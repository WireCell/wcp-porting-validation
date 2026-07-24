# Magnify-tracking-SBND: opening the doc-42 file, and the bugs it exposed (doc 43)

Doc 42 shipped `showcase-stmfit-286241/track_com_286241.root` and told the
reader to open it with `Magnify-tracking-SBND/magnify.sh`.  **Nobody had ever
done that** — the geometry port (`b78b255`) was committed with "visual check
pending (needs X)".  This doc is that check.  The GUI could not compile in
this environment, and once it compiled it aborted on the file; five further
defects showed up in the pictures.  All are fixed here.

## Repro

```bash
# 1. the GUI, headless (needs no X server, writes a PNG per cluster):
cd /nfs/data/1/xqian/toolkit-dev/Magnify-tracking-SBND/scripts
xvfb-run -a -s "-screen 0 1920x1080x24" root -l -q loadClasses.C \
  '/path/to/drive.C("../../wcp-porting-img/sbnd/sbnd_xin/showcase-stmfit-286241/track_com_286241.root","/home/xqian/tmp/shot.png")'
# where drive.C is:
#   void drive(const char* fn, const char* png) {
#       GuiController *gc = new GuiController(gClient->GetRoot(), 1600, 900, fn, 0);
#       gSystem->ProcessEvents(); gc->vw->can->SaveAs(png);
#   }
# interactive equivalent: ./magnify.sh <file>.root   (needs $DISPLAY)

# 2. regenerating the showcase file (source re-run, see B1):
cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin
F="-chord -rescue -rescue-chord -fvz 5 -fvzi 3 -lm -main-pair-real -fvx 2.5 -fvy 3 -stm-fit"
SBND_INPUT_DIR=$PWD/input_files_reco1/extracted-mcp2025c-10evt \
  SBND_WORK_ROOT=$PWD/work-stmbadch ./run_nusel_evt.sh data 8 $F
wire-cell-sbnd-magnify-tracking-convert \
  -bwork-stmbadch/nusel_evt286241/tracking-stm.root -tT_rec_charge \
  -oshowcase-stmfit-286241/track_com_286241.root -f2

# 3. multi-block cross-check (3 tracks in one file):
wire-cell-sbnd-magnify-tracking-convert \
  -bwork-mcp10-stmon/nusel_evt284657/tracking-stm.root -tT_rec_charge \
  -o/home/xqian/tmp/track_com_284657.root -f2
```

## A. Blocker: ACLiC could not compile the GUI here

**Symptom.** `./magnify.sh <file>` dies at `.L Data.cc+` with
`cc1plus: error: /wcwc/stage/root/spack-stage-root-6.32.02-.../include:
Permission denied`, then `ld: cannot find Data_cc_ACLiC_dict.o`.

**Root cause.** ACLiC builds its g++ command by substituting `$IncludePath`
in `gSystem->GetMakeSharedLib()` with `gSystem`'s **and the interpreter's**
include lists.  This spack-built ROOT keeps its own build tree
(`/wcwc/stage/root/spack-stage-.../include`) in the interpreter list.  That
tree is mode 0700 and owned by `wcwc-sync`, and gcc turns "Permission denied"
on an `-I` path into a hard error rather than a warning.

**Why it hid.** The GUI had never been compiled on this machine; the geometry
port was a source-only edit.

**Fix.** `scripts/loadClasses.C` gains `freezeIncludePath()`: it reads both
include lists, drops entries failing `gSystem->AccessPathName(dir,
kReadPermission)`, and substitutes the filtered list into the compile
template before any `.L`.  A no-op where every path is readable.  There is no
API to remove an entry from the interpreter's list, hence the freeze.

**Verification.** All five `.cc` compile; the GUI opens both test files with
zero warnings.

## B. Generation bugs (`tracking-stm.root` → `track_com_*.root`)

### B1. Dead-channel time ranges were garbage — `SbndMagnifyTrackingVisitor`

`Grouping::get_all_dead_chs` converts a dead region's **x range** to ticks, so
a region with unbounded x extent comes back as ±1.3e9.  **84 of 94** T_bad_ch
entries in evt 286241 (93 of 93 in evt 284657) were such values.  With "bad
ch" enabled the GUI drew `TLine(chid, ±3e8, ...)` on every projection pad.

Fixed on both sides:
- visitor clamps to `[0, nticks)` with a new `nticks` config key (C++ default
  3427 = the SBND readout; wired in `cfg/.../sbnd/clus.jsonnet` on the
  `stm_magnify` node only);
- `Data.cc::LoadBadCh` clamps as well and prints how many it clamped, so the
  **already-written** files display correctly.

**Verification.** Re-ran the doc-42 event into a fresh tag `work-stmbadch`
(M13: no existing work root touched; the QL products were symlinked in, so
only the PR step re-ran).  `T_bad_ch` start/end now span `[0, 3427]`;
`T_rec_charge` is unchanged (251 points, identical q sum and rr max) and
`mabc-pr.zip` has the **same content hash** as the record
(`050bed2b…`) — i.e. no physics change.  The committed showcase ROOT was
regenerated from that re-run; every tree except `T_bad_ch` is array-identical
to the previous one.

### B2. The converter dropped `flag_vertex` / `sub_cluster_id` — the GUI aborted

`Data.cc::DrawDQDX` indexes `sub_cluster_id->at(currentCluster)`
unconditionally.  The SBND converter did not carry those two branches (the
uBooNE one does, conditionally), so `SetBranchAddress` failed, the vector
stayed empty, and the GUI died with `terminate called after throwing an
instance of 'std::out_of_range'` **before drawing anything**.

**Why it hid.** My verification of the new app compared the branches the two
apps have *in common* and found them array-identical.  It never asked whether
the uBooNE app writes branches mine does not — which is exactly what happened.

Fixed twice, deliberately: the converter now carries both (guarded by
`GetBranch`, uBooNE-style), and `Data.cc` treats their absence as "one
sub-cluster per track" instead of crashing.

### B3. `Trun` was not carried over

Cosmetic but real: the converted file could not say which run/event it was.
Now cloned.  The GUI does not read it.

## C. Display bugs (`Magnify-tracking-SBND`)

### C1. Every track drew a line to the pad origin

`DrawSubclusters` walked all points with one running cursor: point 0 went to
the sentinel entry (`sub_id[0] = -1`), so every real sub-cluster started one
point late and the **last slot of each TGraph/TPolyLine3D kept its default
(0,0)**.  With a single-segment STM fit the result is a magenta line straight
across the pad from the track to (0,0) — see the difference between the first
two figures below.  Now each sub-cluster is filled from its own recorded
`[start, end]` range.  This was inherited from the uBooNE original and affects
multi-segment PR tracks there too.

### C2. Track markers painted black over the charge they are compared with

The projection pads' sub-cluster graphs set line colour and marker *style* but
not marker *colour*, so ROOT's black default covered the `colz` measured
charge — the one comparison those pads exist for.  Marker colour now follows
the line colour (as the 3D branch already did); same in `DrawProjAll`.

### C3. Default view showed an unreadable speck

A fitted track spans a few hundred of ~4000 channels and ~100 of 857 slices.
New `Data::ZoomToTrack()` frames the six projection pads on the current
track's bounding box (±50 channels, ±30 slices) on every cluster change, and
`GuiController` mirrors that into the Range boxes.  **Judgment call, not a
bug** — UnZoom restores the full plane and the "keep" checkbox still wins.

### C4. Single-entry legend rendered as a full-pad "80"

`TLegend` without an explicit text size scales its entry to the pad.  Fixed
with `SetTextSize(0.04)`.

### C5. `SetRangeUser` warnings on every zoom near an edge

`ZoomProj` (2D and the 3D frame) set ranges past the axis ends — ROOT warned
and silently clamped.  Now clamped explicitly; a full GUI session is
warning-free.

### C6. Control limits too small for real ids

`clusterIdEntry` capped at 1000, but block ids are `cluster_id*10 + pass`
(cluster 100 ⇒ 1000); `pointIndexEntry` capped at 1000 with 554-point tracks
already in hand.  Both raised to 100000; `SetCurrentCluster` still narrows the
point index to the track length.

## D. Findings — NOT fixed, reported

### D1. Rejected-pass blocks have no fitted charge under their own track

For each fitted point, ask whether the block's own `T_proj_data` has a cell
within ±1 channel and ±1 slice:

| event | block | status | points | U | V | W |
|---|---|---|---|---|---|---|
| 286241 | 80 | 0 accepted | 251 | 1.00 | 1.00 | 1.00 |
| 284657 | 140 | 0 accepted | 301 | 1.00 | 1.00 | 1.00 |
| 284657 | 200 | 4 extra-tracks | 114 | 0.08 | 0.24 | 0.24 |
| 284657 | 270 | 2 long-leftover | 139 | 0.18 | 0.39 | 0.29 |

Not a channel-numbering slip: shifting U by ±1984/±3968 channels or the time
by ±1..4 slices recovers nothing, and the missing cells are absent from
*every* block, not merely filed under another id.  The same two blocks carry
~30 % exactly-zero `dQ`.  So on rejected passes the fit's 2D charge
association is degenerate — the same family as the 12.8 % negative-dQ/dx
finding in doc 42 §5.  Accepted STM tracks, the ones the physics rests on, are
clean at 1.00.

### D2. `T_proj_data` can hold a block with no track

evt 284657 has a proj block 180 (cluster 18, 766 cells, 746 of them unique)
with no `T_rec` track: a fitted cell can belong to an associated cluster that
STM never fit, and `write_proj_data` files it under that cluster's id.  Left
as is — the charge is real and the "all clusters" view shows it; the
misleading "should not happen" comment in the visitor is corrected.

### D3. The backward pass has never been exercised

All **36 blocks across the 30 knob-on events have `pass = 0`**.  STM only runs
the backward pass when the cluster is double-ended *and* the forward pass
reaches no decision; that combination did not occur in this sample.  So the
`cluster_id*10 + pass` block-id scheme, and the per-pass duplication of 2D
charge in `write_proj_data`, remain untested on real data.

### D4. One block per event is expected, not lost blocks

Checked because a single-block file tests nothing.  For evt 286241 the tagger
log shows STM evaluating nine clusters and persisting one — `persist_stm_fit`
runs for every evaluated cluster, but only clusters that actually began a pass
have records.  Across 30 events: 36 blocks in 26 events (4 events have none),
8 events multi-block.  Nothing is being dropped.

## E. Gates

- **Compiled-config proof**: knob-off compiled JSON is `cmp`-identical to the
  pre-change tree (`/home/xqian/tmp/magtest/cfg/off_base.json` vs
  `off_new.json`, production op point); knob-on carries `"nticks": 3427` on
  the `SbndMagnifyTrackingVisitor` node.
- **Physics unchanged**: doc-42 event re-run, `mabc-pr.zip` content hash
  identical to the `work-mcp10-stmon` record, `T_rec_charge` array-identical.
- **Not run, and why**: no abtest/qlport A/B.  The converter app and the
  Magnify GUI are outside every pipeline; the visitor edit only executes under
  `save_stm_fit`, which is default-OFF and absent from the compiled config
  when off (proven above).  `wcdoctest` has no `root` package target.

## F. Figures

`showcase-stmfit-286241/magnify_286241_blk80.png` — the GUI on the showcase
file after all fixes: dQ/dx vs L with the Bragg rise (pad 1), the 3D
trajectory (pad 3), the fit over measured charge per plane (pads 4–6) and the
`(pred−meas)/meas` residuals (pads 7–9).
