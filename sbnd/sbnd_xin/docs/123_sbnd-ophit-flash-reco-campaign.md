# doc 123 — SBND flashes from OpHits: investigation and campaign design

2026-09-24.

**Status: design only.** No toolkit, sbndcode, or colleague code is changed. The only new
things are two read-only probe scripts and their output tables, listed in §9.

**Question.** SBND Q/L matching uses the reco1 `recob::OpFlash`. That flash merges or drops
flashes that are a few µs apart. Can we rebuild the flashes from the reco1 `recob::OpHit`,
the way the toolkit light reco does for PDHD/PDVD? If so, how should the new flashes enter
Q/L matching, given the geometric patches we already added to save clusters?

## Repro block

The first-hand numbers in §1 and §6 come from:

```sh
cd sbnd_xin
S=scripts/analysis/light
D2=/nfs/data/1/xning/wirecell-working/SBND/data/reco1_upstream/data_reco1      # mcp2k (symlinks into yuhw's area)
cut -d: -f1 docs/123_flash/123_rescue_census_detail.txt > ~/tmp/d123_events40.txt

# OpHits and OpFlashes for the 40 rescue-fired events, bare ROOT with the wire-cell-sbnd-reco1 mirror dictionary
root -l -b -q "$S/d123_ophit_probe.C(\"input_files_reco1/data_MCP2025C_reco1_frameshift_first1000ev.root\",\"$HOME/tmp/d123_events40.txt\",\"$HOME/tmp/mcp1k.tsv\",-20,30)"
for p in 1 2; do
  root -l -b -q "$S/d123_ophit_probe.C(\"$D2/data_MCP2025C_reco1_frameshift_2nd1k_part$p.root\",\"$HOME/tmp/d123_events40.txt\",\"$HOME/tmp/mcp2k_p$p.tsv\",-20,30)"
done

python3 $S/d123_rescue_vs_hits.py ~/tmp/{mcp1k,mcp2k_p1,mcp2k_p2}.tsv docs/123_flash/123_rescue_census_detail.txt \
    > docs/123_flash/123_rescue_vs_hits.tsv
python3 $S/d123_summary.py docs/123_flash/123_rescue_vs_hits.tsv ~/tmp/{mcp1k,mcp2k_p1,mcp2k_p2}.tsv \
    > docs/123_flash/123_rescue_onsets.tsv
python3 $S/d123_null.py /nfs/data/1/xning/wirecell-working/SBND/light/flash_from_hits/config/sbnd-pmt-channels.json \
    docs/123_flash/123_rescue_onsets.tsv ~/tmp/{mcp1k,mcp2k_p1,mcp2k_p2}.tsv > docs/123_flash/123_null_onsets.tsv
```

The probe scans `EventAuxiliary` over every entry; each 4 GB mcp2k file takes a few minutes.
All 40 events were found: 11 in mcp1k, 20 in mcp2k part1 and 9 in mcp2k part2.

**Where each claim comes from.** Every claim below carries one of these tags:

- **[V]** — I checked it first-hand this session: I ran the command or read the source line.
- **[C]** — xning reports it in `/nfs/data/1/xning/wirecell-working/SBND/light/*.md`. I have
  not re-run it.
- **[S]** — read from source code or fcl by the investigation (file and line given), without
  running anything.
- **[D]** — from an earlier doc in this tree (doc number given).

---

## 0. Answers in brief

| # | question | answer |
|---|---|---|
| 1 | Do reco1 files carry OpHits? | **Yes.** `recob::OpHits_ophitpmt__Reco1.` and the `OpFlash↔OpHit` Assns are in every SBND reco1 sample we use. PMT hits only: no waveforms, no X-ARAPUCA. **[V]** |
| 2 | How does LArSoft build OpFlash from OpHit? | `SBNDFlashFinder` + `SimpleFlashAlgo`, run separately per TPC. It takes the brightest 10 ns seed first, integrates **8 µs**, then vetoes **±8 µs** around it. So two flashes less than 8 µs apart cannot both survive in one TPC. **[V]** for fcl and code lines. |
| 3 | How does the toolkit separate close flashes? | `flash/OpFlashFinder` is a larana `OpFlashAlg` port: 1 µs dual accumulators, then a per-hit **width-tolerance refinement** (the step that separates), then late-light removal. For SBND, xning's `SBNDOpFlashFinder` adds a **hit-level pulse split**. **[S]**, **[C]** |
| 4 | Design | Use xning's `SBNDOpFlashFinder`, fed by an OpHit reader in `wire-cell-sbnd-reco1`. It writes the same `[nflash, 313]` opflash tensor, so it is a drop-in for the existing reader. Bring it in behind a default-OFF knob (§4). |
| 5 | Integration with Q/L and the geometric patches | Keep the cathode rescue ON while the new flashes are evaluated. Measure how often it still fires; that residual measures the inefficiency the new flashes leave behind. Only then decide whether to retire rescue knobs (§5). **The production path runs inside LArSoft and has no OpHit source: this is an owner decision (§5.1).** |
| 6 | Close-flash examples | The cathode-rescue census has 40 events and 41 moves. 22 moves have Δt0 inside the 8 µs veto; in **21 of them** the TPC without a reco1 flash has a clear hit-level pulse at that time (≥100 PE, ≥3× the preceding 200 ns). The same rule fires on **0 of 166** null probes 0.8–1.5 µs away. Three more moves (72759, 78242, 317427) have Δt0 outside the veto but still show the veto signature. The OpHits hold what reco1 threw away. **[V]** (§6) |

---

## 1. Reco1 files contain the OpHits

Branches, listed with uproot from the `Events` tree **[V]**:

```
recob::OpHits_ophitpmt__Reco1.
recob::OpFlashs_opflashtpc0__Reco1.            recob::OpFlashs_opflashtpc1__Reco1.
recob::OpFlashrecob::OpHitvoidart::Assns_opflashtpc{0,1}__Reco1.
```

| sample (our tag) | file | entries | OpHit branch |
|---|---|---|---|
| nueCC48 / fsprod | `input_files_reco1/data_filtered_decoded_reco1-fe6033f3-…_eventidfiltered_frameshift.root` | 48 | yes **[V]** |
| mcp1k | `input_files_reco1/data_MCP2025C_reco1_frameshift_first1000ev.root` → yuhw `round2-patrec/` | 1000 | yes **[V]** |
| mcp2k | `…/data_MCP2025C_reco1_frameshift_2nd1k_part{1,2}.root` (doc 102 sample table; xning's `data_reco1/` links to the same files) | 2×1000 | yes **[V]** (xning also checked) |
| MC | xning `reco1_upstream/mc_reco1/reco1-detsim-g4-gen-Gen2_2026-*.root`, 10 files | 104 | yes **[C]** (branch list), not re-checked here |

The `.obj` is stored unsplit, so uproot cannot read the hits. Bare ROOT with the mirror
dictionary already in `wire-cell-sbnd-reco1` can. That dictionary has carried
`recob::OpHit` v15 since the initial import (`inc/WireCellRoot/SBNDReco1Products.h:94`,
`dict/LinkDef.h:22-44`). `d123_ophit_probe.C` reads the hits **[V]**: 12.7–53.5 k hits per
event on the 40 events (for example 28 657 on evt 288952). Its field is
`t = fStartTime + fRiseTime`, the same input time `SBNDFlashFinder` uses
(`OpHitInputTime "RiseTime"`).

**What the hits are.** Most are single photoelectrons: the median width is 0.034 µs **[C]**.
The exceptions are the brightest PMTs of a large flash, where one hit can be up to 5.8 µs wide
and hold thousands of PE **[C]**.

Upstream they come from Wiener deconvolution (`opdecopmt`) followed by larana
`SlidingWindow` hit finding, with `SPEArea 200`, `HitThreshold 1` and per-channel ADC
thresholds **[S]** (`sbndcode/OpDetReco/OpDeconvolution/job/sbnd_ophitfinder_deco.fcl:72-105`).
23 PMT channels are masked in `ophit_finder_sbnd.fcl:192` **[S]**.

We can re-find flashes from the hits. We cannot re-find the hits themselves: there are no
waveforms in reco1.

## 2. How LArSoft turns OpHits into the OpFlash we use

The chain is `ophitpmt` → `opflashtpc0` and `opflashtpc1`. The module is `SBNDFlashFinder`
and the algorithm is `SimpleFlashAlgo` (the only one registered). Data: `reco1_data.fcl:25-32`.
MC: `workflow_reco1.fcl:21-32` **[S]**.

Parameters, from `sbndcode/OpDetReco/OpFlash/job/sbnd_flashalgo.fcl:5-10` **[V]**:

```
PEThreshold 20   MinPECoinc 6   MinMultCoinc 3   IntegralTime 8.   PreSample 0.1   VetoSize 8.   (µs)
```

**How `SimpleFlashAlgo` builds flashes** (`FlashFinder/SimpleFlashAlgo.cxx`):

1. **Per TPC.** The PMTs are split by OpDet x < 0 or x > 0, and each TPC is clustered on its
   own. The two collections are never reconciled with each other **[S]**.
2. **Seeds.** Hits are put in 10 ns bins. A bin is a seed if it has ≥ 6 PE and ≥ 3 hits.
   Seeds are processed **brightest single bin first**; the map is keyed by `1/pe` **[S]**.
3. **Veto.** A candidate whose start is within ±8 µs of an already accepted flash is skipped.
   These are the two `skip=true` branches at about L224 and L228 **[V]**.
4. **Accept.** The flash integrates PE and hits over `[seed − 0.1 µs, seed + 7.9 µs)` and is
   kept if that sum is ≥ 20 PE.
5. **Time.** The time is the seed bin. `SBNDFlashFinder_module.cc:~164-178` then refines it
   with `FlashT0SelectedChannels` and subtracts a light-propagation time taken from a PMT-ratio
   drift estimate **[V]**.

**So two real flashes Δt < 8 µs apart in one TPC become one flash** **[S]**:

- **The dimmer one comes later.** It is vetoed, and its PE falls inside the brighter flash's
  window. Its PE and per-PMT pattern are added to the brighter flash.
- **The dimmer one comes earlier.** It is vetoed, and its light is before the other flash's
  `−0.1 µs` pre-sample, so it is in no flash at all.

Because each TPC keeps whichever flash is brighter *in that TPC*, the two TPCs can keep
different members of the pair. This is the evt 59415 signature in
`toolkit/match/docs/sbnd-opdetreco-chain.md` §"Known failure mode": the pair is 4.79 µs apart
and each flash is missing on the other side. This doc agrees with that analysis and does not
redo it.

**Two more reco1 defects xning found.** Any comparison has to keep these separate from the
close-flash effect (§7 R2):

- **The 74 µs time bug** **[C]**. When the drift estimate fails it returns −999999. The
  propagation correction then shifts `fTime` 74.06 µs late. This affects 9.9 % of MC flashes
  and 0.59 % of data flashes, almost all small. On 14 MC events it moved 13 of 204 Q/L pairs.
  (`FLASH_DRIFT_POSITION.md`, `QL_MATCHING_IMPACT.md`.)
- **Dropped bright flashes** **[C]**. On a bright PMT, one giant hit can fail the ≥ 3-hits
  seed rule. A later, dimmer flash then vetoes the bright one. About 97 k PE was lost over 13
  MC events. xning's Python emulator of `SimpleFlashAlgo` reproduces all 449 MC and 94 data
  reco1 flashes exactly (`FLASH_FROM_HITS.md` §"SBND loses some bright flashes").

**Could LArSoft itself do better?** Two options exist, and neither is attractive **[S]**:

- **Shrink `VetoSize` and `IntegralTime` to 2–3 µs.** Every flash's PE integration changes,
  and `SimpleFlashAlgo` has no late-light rejection, so a bright flash's slow tail would seed
  false flashes.
- **Use larana `OpFlashFinder`.** It is configured (`opticaldetectormodules_sbnd.fcl:250`) but
  commented out of reco1. It has no per-TPC split, no t0 tool and no propagation tool, and it
  is not tuned for deconvolved SBND hits.

Either option is a change to the SBND production reco1, which we do not own. Hence the
toolkit route.

## 3. How the toolkit separates close flashes (PDHD/PDVD), and xning's SBND version

**The PDHD/PDVD chain** **[S]**, documented in `toolkit/flash/docs/design.md` and
`stage2-reconstruction.md`:

```
OpWaveformSource → OpDecon → OpHitFinder → [OpHitMerge] → OpFlashFinder → TensorFileSink
                                                          "opflash" [nflash, 1+nchan] + "flash_summary" + "ophits"
```

`flash/src/OpFlashFinder.cxx` ports larana `OpFlashAlg::RunFlashFinder`:

1. **Two accumulators.** The bins are 1 µs wide (`bin_width`), and the second set is offset
   by half a bin. A bin becomes a candidate when it crosses `flash_threshold`.
2. **Claim.** Candidates are processed largest first, and each claims the hits no other
   candidate has taken.
3. **Width refinement** (cxx:77-137). Inside each claimed group, a sub-flash grows from the
   largest hit. A hit joins it if `|t_hit − t_flash| ≤ 0.5·(half-width_hit + half-width_flash)`.
   **This is the step that separates two flashes inside one bin.**
4. **Late-light removal.** A later flash j is dropped if
   `(PE_j − hyp)/√hyp < 3`, with `hyp = PE_i·(w_j/w_i)·e^{−Δt/1.6 µs}`.
5. **Optional re-merges.** `flash_refine` (PDHD) and `flash_tail_merge` (PDVD production)
   merge back over-split tails.

There is no per-TPC veto. Flash time is the PE-weighted mean of hit peak times. The output
tensor feeds `aux/src/FlashTensorToOpticalPCs.cxx`, which feeds `QLMatching`.

**The SBND version: `SBNDOpFlashFinder`** **[C]**, plus a code read.

- **Where it lives.** All of it is **uncommitted** in xning's trees:
  - toolkit: `/nfs/data/1/xning/wirecell-working/toolkit/flash/{inc/WireCellFlash/SBNDOpFlashFinder.h, src/SBNDOpFlashFinder.cxx, test/doctest_sbndopflashfinder.cxx}`,
    on `apply-pointcloud` at b5a897f6;
  - reader: `SBNDReco1OpHitSource` and `SBNDReco1CafOffset.h`, plus edits to
    `SBNDReco1OpFlashSource.cxx`, in xning's `wire-cell-sbnd-reco1`;
  - jobs: `SBND/light/flash_from_hits/wct-reco1-flash-from-hits.jsonnet` and
    `SBND/light/wct-ql-from-hits.jsonnet`.
- **The algorithm, in order:**
  1. Two accumulators of 8 µs bins, offset by half a bin, 20 PE threshold. This deliberately
     reproduces SBND's grouping scale.
  2. Claim, as in the toolkit.
  3. **Pulse split** inside a claimed group. The group is scanned in 10 ns bins. A later
     spike of ≥ max(30 PE, 1 %) cuts the group when all of these hold:
     - it is ≥ 0.5 µs after the onset;
     - its 100 ns burst, with the brightest OpDet removed, is ≥ max(100 PE, 5 % of the
       onset burst);
     - it comes after a dip: the 500 ns before it hold ≤ 50 % of the burst.
  4. **Join**: over-split pieces are merged back.
  5. Late-light removal, as larana does it.
  6. At least 3 fired PDs.
  7. **Time**: SBND's prompt rule (top hits within −20/+10 ns of the brightest 10 ns bin,
     until 60 % of its PE).
- **What it leaves out.** There is no X estimate and no light-propagation correction, so the
  74 µs bug cannot happen, but times sit about 5–8 ns off reco1.
- **Output.** The same `opflash [nflash, 313]` tensor as `SBNDReco1OpFlashSource`, so
  `TensorFileSource → FlashTensorToOpticalPCs → QLMatching` reads it unchanged.

**xning's results** **[C]**, not re-run here:

| | MC (13 evt) | data (63 evt) |
|---|---|---|
| reco1 flashes recovered | 449/449 | 1782/1782 |
| one-to-one | 97 % | 91 % |
| PE ratio, median | 1.000 | 1.005 |
| extra flashes | 142 (median 36 PE) | **723 (median 51 PE)** |
| MC truth: flashes still holding a second pulse ≥ 20 PE | 13/589 (reco1: 17/405) | — |
| Q/L, MC: same flash as bug-fixed reco1 | 163/185 clusters | not run |

Their open items: no tail subtraction after a split, so the later piece carries 1.5–2× its
true PE; small second pulses on a bright tail are still merged; no Q/L on data yet.

## 4. Design: a new SBND flash reco

**Principle.** Reproduce reco1 first, then change one thing at a time. The new source must be
a drop-in producer of the existing `opflash_apa<N>` tensor sets, so nothing downstream of
`TensorFileSource` changes.

```
SBND reco1 art file
  └─ SBNDReco1OpHitSource  (wire-cell-sbnd-reco1, per TPC: channels = even/odd PMT OpChannels)
       └─ SBNDOpFlashFinder (toolkit flash/)  → opflash [nflash,313] (+ flash_summary, ophits with flash id)
            └─ md: run/subrun/event + frame_apply_at_caf       ← MUST be carried (see below)
                 └─ TensorFileSink opflash_apa{0,1}.tar.gz  ─→ unchanged run_ql_evt.sh / wct-clus-matching-perevt.jsonnet
```

- **Where the code goes.**
  - The OpHit reader belongs in `wire-cell-sbnd-reco1`, **not** toolkit `root/`. Its mirror
    dictionary clashes with LArSoft's (`wire-cell-sbnd-reco1/docs/DESIGN.md`, WCT issue #494).
  - `SBNDOpFlashFinder` in toolkit `flash/` is allowed. It depends only on `flash` and `aux`
    and adds no external dependency.
- **The flash-time offset is an acceptance item.**
  - Today `SBNDReco1OpFlashSource` writes `frame_apply_at_caf` into the tensor-set metadata,
    and `FlashTensorToOpticalPCs` adds it to every flash time (doc 21). It is 0.25–2.7 µs per
    event.
  - The new chain must carry it through the hit tensor into the flash tensor. xning's finder
    writes `offset_us` only to metadata and does not apply it. Check whether it passes
    `frame_apply_at_caf` through.
  - If the offset is dropped, every flash moves out of the 0.3–1.9 µs beam window and out of
    the rescue's 0.2–2.2 µs cut, with no warning.
  - Test: on the 48-event fsprod sample, at least 45 of 48 in-time flashes must sit in
    +0.3…1.9 µs, the doc 21 reference.
- **Per-TPC split.** Keep one flash collection per TPC, as reco1 does, so APA ports, gid
  offsets (+1000000 for APA1) and `flash_group_window` (80 ns) keep their meaning.
  - PMT parity: even OpChannel = TPC0, odd = TPC1. This comes from xning's
    `sbnd-pmt-channels.json` (60 + 60 PMTs), and the probe uses it. **[V]**: all 97 channels
    with hits in the 40 dumped events (179 932 hits; 120 PMTs minus the 23 masked) are in
    that list, 50 even in `tpc0` and 47 odd in `tpc1`, with no violations (`d123_null.py`).
  - A joint two-TPC finder, which would fix the cross-TPC disagreement in one step, is an
    option for a later round, not round 0.
- **Knob.** A default-OFF switch in the SBND job selects "reco1 OpFlash" (today) or "hit
  flashes" (new). Use key suppression so the compiled production config is byte-identical
  when off (§2 of the operating manual).

## 5. Integration with Q/L matching and the geometric patches

**What exists today (ON in SBND production):**

- **`ClusteringCathodeBundleRescue`** (`clus/src/clustering_cathode_bundle_rescue.cxx`; docs
  pr/14, pr/17, 73) **[D]**.
  - It takes a beam-window cluster with a tip within 5 cm of the cathode. It pairs it with a
    cluster on the other side that sits on a different flash with
    **Δt0 ∈ [−8, +13] µs**, which is essentially the reco1 veto. It also pairs it with a
    cluster that has no flash (`rescue_unmatched`).
  - It merges the pair, then an a/b/c/d length rule chooses which flash the merged object
    takes.
  - Round 2 and 3 knobs (`rescue_geom_first`, `rescue_pierce_test`, `rescue_allow_in_beam_far`, …)
    extend it to wrong-flash cases +28…+855 µs away.
- **`nu_bundle_flash_group`** (doc 109 r4) **[D]**. It merges neutrino bundles whose two TPC
  flashes fall in one 80 ns group and whose charge touches within 20 cm. Both flashes exist
  here, so it is not a flash-inefficiency patch.
- **`QLMatching` cross-TPC checks** **[D]**. `xtpc_flag` and `flash_group_window` 80 ns only
  pair flashes whose times agree to 80 ns. A pair split by the veto is never looked at as a
  pair (`match/docs/qlmatching-evt59415-xtpc-flash-split.md`).

**How the new flashes should meet these patches:**

1. **Rescue stays ON first.** Run the new flashes with production rescue on.
   - Count how often rescue still fires, and on which moves. Once hit flashes restore the
     partner flash, the 22 within-veto moves (§6) should stop firing: both halves find
     time-coincident flashes and `xtpc`/grouping pairs them.
   - Whatever still fires measures the inefficiency the new finder leaves.
2. **Only then try rescue OFF**, or retire individual rescue knobs, on the same events. The
   moves that are *not* flash losses must be checked separately, because they belong to the
   patch's other duties:
   - same-time moves;
   - 8–13 µs moves;
   - geom-first moves beyond 13 µs.
3. **Watch the extra small flashes.** In data, xning's finder adds about 11 flashes per event
   **[C]**, with a median of 51 PE, right at `flash_minPE` 50. That means many new candidates
   in the global LASSO fit. Measure how many clusters move to a < 100 PE flash before trusting
   any efficiency gain.
4. **Beam-flash PE changes.** Once a merged partner is split off, the beam flash loses the
   partner's PE. Q/L scores (KS, χ², over-prediction veto) then compare against a different
   PE pattern. Split-off tails also over-count by 1.5–2× **[C]**. Expect movers even on
   clean events. Adjudicate them by hand scan, not by the matched count.

### 5.1 The owner decision that gates production

SBND production runs Wire-Cell **inside LArSoft**:
`cfg/pgrapher/experiment/sbnd/wcls-img-clus-matching-xin.jsonnet:66-79` reads
`recob::OpFlash` through `wclsOpFlashSource`. **No wcls OpHit source exists.** I did not find
one in the local sbndcode, or in larwirecell v10_04_03 on cvmfs.

The standalone reco1 path is enough for development and for this whole campaign. To use hit
flashes in production, one of these is needed:

- **(a)** a new larwirecell OpHit source (`recob::OpHit` → the `ophits` tensor), with the same
  finder running in the wcls graph;
- **(b)** an sbndcode-side change: new reco1 flash settings, or a new flash module, which SBND
  owns;
- **(c)** keep hit flashes as a standalone-only improvement.

This does not block rounds R0–R4 below, but it decides where the campaign ends.

## 6. Close-flash examples from the cathode-rescue census

**Where the cases come from.** The census ran with all rescue knobs on over mcp1k + mcp2k
(3000 events, doc 73 / cbr3). It fired on 40 events and 41 moves; 56463 has both a rescue and
an unmatched-rescue move.

- The per-move log is now committed as `docs/123_flash/123_rescue_census_detail.txt`. It was
  only in `~/tmp/cbr3/`, which cleanup rounds sweep.
- For each move, the probe found the two reco1 flashes the halves were matched to, using the
  rescue's Δt0, which does not depend on the flash-time offset. The 39 two-flash moves were all
  matched this way within 30 ns. The 2 unmatched moves have no far flash, so for those the
  probe takes the near-side in-time flash with the most PE.

**The test.** For each move, look at each TPC at the time where it has no reco1 flash:

- A = far TPC at the near (beam-side) time;
- B = near TPC at the far time.

Sum the hit PE in `[t − 50, t + 150] ns` ("prompt") and in `[t − 350, t − 150] ns` ("pre",
same width). An **onset** is prompt ≥ 100 PE and prompt ≥ 3 × pre, i.e. a new pulse, not the
tail of an earlier flash.

Full tables: `docs/123_flash/123_rescue_vs_hits.tsv` and `123_rescue_onsets.tsv`. **[V]**

| class (by Δt0) | moves | reading |
|---|---|---|
| **within the 8 µs veto**, 0.05 < \|Δt0\| ≤ 8 µs | **22** | The veto signature: the near TPC's nearest reco1 flash to the far time is always the beam-side flash itself, at −Δt0 (22/22). The far TPC's nearest flash to the beam time is the partner at +Δt0 in 20/22; in 169824 and 288952 it is another flash 3.6 and 2.1 µs away. **21/22 show a hit-level onset** in a flashless TPC. The exception, 173450, has A = 301 PE with pre = 239. **These are the target of the new finder.** |
| far half unmatched (`rescue_unmatched`) | 2 | 50801: A = 110 PE (pre 4). 56463: A = 20 996 PE (pre 3), yet the far TPC's nearest reco1 flash is 4.56 µs away. Both are onsets, so both are hit-recoverable. |
| same time, \|Δt0\| < 0.05 µs | 8 | Both flashes exist in both TPCs. The rescue is doing cathode-crossing charge merging, not recovering light. Expected to keep firing. |
| 8–13 µs | 4 | In 164576, 397498 and 407798 both TPCs have a reco1 flash at both times, within 26 ns: a Q/L choice problem, not light loss. **72759 is a veto loss:** the far TPC's nearest reco1 flash to the beam time is 0.60 µs away, and there is a 124 PE onset (pre 7) at the beam time. |
| geom-first, > 13 µs | 5 | 65053, 78242, 281165, 317427, 319913. Three have a far-TPC reco1 flash at the beam time (within 0.1 µs; 65053's is 92 ns off, *outside* the 80 ns grouping window). **78242 and 317427 have no far-TPC beam flash:** the nearest is 2.05 and −2.39 µs away, with 3384 and 2855 PE onsets at the beam time. They are veto losses that Q/L then routed to a wrong flash hundreds of µs away. |

**Examples to use** (all run 18255/18259, data). *Missing* means the TPC and time where the
onset has no reco1 flash:

| evt | sample, entry | Δt0 (µs) | missing | onset PE (pre) | note |
|---|---|---|---|---|---|
| 56463 | mcp1k 599 | +4.56 | far TPC1 @ beam | 20 996 (3) | also the pr/17 founding event; the biggest loss |
| 78369 | mcp2k_p1 490 | +3.01 | near TPC0 @ far t | 23 648 (63) | beam flash 95 k PE absorbed the partner |
| 82608 | mcp2k_p1 645 | +3.79 | near TPC0 @ far t | 13 570 (13) | |
| 395148 | mcp1k 472 | −1.46 | near TPC0 @ far t | 16 399 (2) | earlier partner, so its light is **lost**, not absorbed |
| 169824 | mcp1k 809 | −4.47 | both TPCs | 3481 / 12 319 | each TPC kept a different member: the 59415 pattern |
| 179369 | mcp2k_p1 140 | +1.67 | both | 8277 / 2962 | |
| 288952 | mcp1k 227 | −6.50 | both | 1653 / 245 | pr/14 hand-scanned move |
| 392200 | mcp1k 453 | +2.43 | both | 140 / 2593 | pr/14 hand-scanned move |
| 78242 | mcp2k_p1 697 | +855 (geom) | far TPC1 @ beam | 3384 (0) | veto loss hidden behind geom-first |
| 70538 | mcp2k_p1 862 | +0.002 | — | — | **negative control:** same-time, both flashes exist |

**Null floor** **[V]**. Is 21/22 more than chance, given that the test times sit a few µs from
a bright flash in a TPC that is lit?

- For each of the 22 within-veto moves, `d123_null.py` re-runs the same onset rule on the same
  TPC at t_miss + δ, for δ ∈ {−1.5, −0.8, +0.8, +1.5} µs, on both sides.
- It skips any δ that lands within 0.3 µs of a reco1 flash on that side.
- Result: **0 of 166 null probes** fire, and 0 of 22 moves have any null onset
  (`docs/123_flash/123_null_onsets.tsv`).
- So the onsets at the vetoed times are real pulses, not the rule firing on a lit TPC.

**Caveats on this test.**

- The 100 PE / 3× thresholds are my choice. They were not tuned. A 200 ns box is cruder than
  xning's split rule.
- The pre-window control guards against a smooth tail, not against a second bright pulse just
  before t.
- The probe dumps hits in −20…+30 µs raw only. For the geom-first moves, the far time is
  outside that range, so their side-B prompt PE of 0 is an artefact of the window, not a
  measurement.
- The census sample is **biased**. It only sees close-flash cases whose charge crosses the
  cathode near the beam window. It says nothing about how often close flashes occur, or about
  close flashes inside one TPC with no cathode-crossing charge. That needs the unbiased census
  in R1.

---

## 7. Campaign rounds (for later sessions)

Each round follows the operating manual:

- every behaviour change is default-OFF;
- the knob-off config and outputs are byte-identical under `hash_archive.py` gates, with the
  labels reported;
- labels and tags are new, never overwritten.

**R0 — bring-in, no behaviour change.**

- xning commits `SBNDOpFlashFinder` (toolkit `flash/`, with their doctests) and
  `SBNDReco1OpHitSource` (`wire-cell-sbnd-reco1`). It is their code, so they author it.
- Add a runner or dump option to write hit-flash `opflash_apa{0,1}.tar.gz` into a *new*
  extracted tag.
- Add a default-OFF `flash_source` knob in the SBND per-event job.
- Gates:
  - knob-off compiled JSON byte-identical;
  - SBND QL gate (the evt 686 style of doc 21) member-hash identical;
  - `wcdoctest-flash` passes;
  - freshness proof on `libWireCellFlash.so`.
- Acceptance for the flash-time offset: at least 45 of 48 in-window on fsprod (§4).
- Resolve xning's open item that the Sep-18 `local/lib` install lacks the class (M1).

**R1 — unbiased close-flash census on the hits.**

- For every reco1 flash in mcp1k + mcp2k, and in the MC files with truth, scan its own
  associated hits (the Assns branch) for a second onset, using the §6 test and xning's split
  rule.
- Output: the rate and PE of second pulses versus Δt, per TPC and in both TPCs; the fraction
  of beam-window flashes carrying a partner; and cross-TPC disagreement (59415-type).
- Check that the hit-level second-pulse time reproduces the rescue Δt0 on the 22 + 2 moves.
- MC truth gives efficiency and purity of the split.

**R2 — attribution arms, so a gain is credited to the right fix.** The new finder changes
three things at once: it splits close flashes, it removes the 74 µs bug, and it keeps bright
flashes that reco1 drops. Arms, all from hits:

| arm | contents | purpose |
|---|---|---|
| A0 | xning's emulator at VetoSize 8 (≡ reco1, exact per §2) | null |
| A1 | A0 + 74 µs fix | isolates the bug fix |
| A2 | A1 + keep bright flashes | isolates the seed-rule fix |
| A3 | `SBNDOpFlashFinder` | adds the splitting |

- Run A0 against the stored reco1 OpFlash first; this is the null pair and it must be exact.
- Run Q/L on each arm, on nueCC48 plus a mcp subset.

**R3 — Q/L with the new flashes, rescue ON.**

- Full mcp1k + mcp2k + nueCC48 through the existing chain, on a new work root.
- Measure:
  - the rescue-firing census against the 41 moves in §6 (the prediction: the 22 within-veto moves, the 2
    unmatched moves and the 3 veto losses behind 72759, 78242 and 317427 stop firing);
  - neutrino-candidate count, and the beam flash per event;
  - movers, split by whether the matched flash is under 100 PE.
- Hand-scan the movers blind, with a fresh tag.

**R4 — the rescue decision.**

- On the R3 hit-flash arm, run rescue OFF, and each round-2/3 knob OFF in turn.
- Keep what still earns its place (same-time merging, geom-first); retire what the hit flashes
  make redundant.
- Also look at the 65053-type 92 ns offsets, where the partner exists but misses the 80 ns
  grouping window.
- Owner flip decision.

**R5 — the production path** (§5.1, owner decision). A larwirecell OpHit source, an SBND reco1
change, or standalone only.

**Later options, not scheduled:**

- a joint two-TPC finder, which fixes cross-TPC disagreement at the source;
- tail subtraction after a split, which fixes the 1.5–2× PE on split-off pieces;
- a MicroBooNE-style per-PMT KS pattern test to separate close flashes
  (`prototype_base/2dtoy/src/ToyLightReco.cxx`, per `sbnd-opdetreco-chain.md`).

## 8. Open questions for the owner

1. The production path for hit flashes, §5.1: (a), (b) or (c).
2. Bring-in form. Is xning's code accepted as the SBND variant in toolkit `flash/`, next to
   `OpFlashFinder`? Or should the existing `OpFlashFinder` get SBND knobs? The manual prefers
   duplication over in-place generalisation.
3. Which extra small flashes are acceptable in Q/L: keep `flash_minPE` 50, or raise it for hit
   flashes only.

## 9. Files

- `docs/123_sbnd-ophit-flash-reco-campaign.md` — this doc.
- `docs/123_flash/123_rescue_census_detail.txt` — copy of `~/tmp/cbr3/census-firing-detail.txt`:
  40 events, all rescue knobs on, mcp1k + mcp2k.
- `docs/123_flash/123_rescue_vs_hits.tsv` — per move: sample, entry, Δt0, class, matched reco1
  flashes, prompt PE, and the nearest reco1 flash on each side.
- `docs/123_flash/123_rescue_onsets.tsv` — per move: the onset test on sides A and B, and
  whether a hit-level split is expected.
- `scripts/analysis/light/d123_ophit_probe.C` — bare-ROOT OpHit/OpFlash dump. It needs
  `wire-cell-sbnd-reco1/install/lib/libWireCellSBNDReco1.so`.
- `scripts/analysis/light/d123_rescue_vs_hits.py`, `d123_summary.py` — the §6 analysis.
- `scripts/analysis/light/d123_null.py`, `docs/123_flash/123_null_onsets.tsv` — the §6 null
  floor and the PMT parity check.
- Raw hit dumps (6.8 MB) are not committed; the Repro block regenerates them.

**External references, read-only:**

- xning's docs: `/nfs/data/1/xning/wirecell-working/SBND/light/{README,FLASH_FROM_HITS,FLASH_DRIFT_POSITION,MC_FLASH_STUDY,QL_MATCHING_IMPACT}.md`;
- `toolkit/match/docs/sbnd-opdetreco-chain.md`, `qlmatching-evt59415-xtpc-flash-split.md`;
- `toolkit/root/docs/sbnd-reco1-source.md`; `wire-cell-sbnd-reco1/docs/DESIGN.md`;
- sbndcode `OpDetReco/OpFlash/{job/sbnd_flashalgo.fcl, FlashFinder/SimpleFlashAlgo.cxx, SBNDFlashFinder_module.cc}`.
