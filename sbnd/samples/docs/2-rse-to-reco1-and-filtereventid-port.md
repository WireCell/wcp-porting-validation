# 2 – RSE → reco1 files (samweb) + FilterEventID port to larwirecell

Task (2026-07-13): starting from lynnt's nuecc-candidate `*_eventidfiltered.root`
decoded files, (1) find the matching **reco1** artROOT files by run/subRun/event,
and (2) port sbndcode's `FilterEventID` module into larwirecell and re-filter the
reco1 files down to just those events.

Work dir: `/exp/sbnd/app/users/yuhw/wcp-porting-img/sbnd/samples/`.
Input RSE list: `lynn-nuecc-rse.csv` (48 events — see doc `1-…`).

---

## Step 1 — RSE → reco1 files via samweb

Script: `find_reco1_files.sh <DEF> <RSE_CSV> <OUT_FILES.lst> <OUT_MISSING.lst>`

Key facts:
- **samweb has NO `event_number` dimension.** The queryable match key is the file
  metadata field **`sbnd.event_number_list`** (an underscore-delimited list of the
  event numbers stored in each reco1 file).
- reco1 files hold **many** events each (not 1/file); the list enumerates them.
- Correct query per RSE:
  ```
  defname:<DEF> and run_number <R> and sbnd.event_number_list %_<EVT>_%
  ```
  The `%_..._%` wildcard is **required** — a bare number matches nothing. The `_`
  delimiters prevent false substrings (`_302_` does not match inside `_3021_`).
  Verified to match first / middle / last events in a multi-event list.
- Physical path resolution: `samweb locate-file <file>` → grep the `/pnfs…` path;
  (xrootd URL alternative: `samweb get-file-access-url --schema=root <file>`).
- samweb env: SL7 apptainer → `setup sam_web_client; export SAM_EXPERIMENT=sbnd`.

Definitions used (BNBLight **data** reco1, MCP2025C, v10_14_02):
- primary  `data_MCP2025C_Fall25-Run1_BNB_FixedDev_bnblight_v10_14_02_reco1_sbnd`
- fallback `data_MCP2025C_Fall25-Run1_BNB_RollingDev_bnblight_v10_14_02_reco1_sbnd`

Coverage note: FixedDev only contains runs **18255** and **18259**; RollingDev
covers the other runs. For the 48-event list: 37 found in FixedDev + 11 recovered
from RollingDev = **48 found, 0 missing**. (The event-run numbers 18253…18409 in
the CSV differ from the filename runs — match on the CSV run, not the filename.)

Deliverables (`samples/`):
- `lynn-nuecc-reco1-files.lst`      — 47 unique reco1 /pnfs paths
- `lynn-nuecc-reco1-missing-rse.lst`— empty (0 missing)
- `lynn-nuecc-reco1-files.map.txt`  — 48 RSE→file audit mappings

---

## Step 2 — FilterEventID ported to larwirecell

Source: sbndcode branch `feature/lynnt_evtfilter`,
`sbndcode/Commissioning/FilterEventID_module.cc` (+ `fcls/filtereventid.fcl`).
Simple `art::EDFilter` (namespace `filt`): keeps an event when paired
`filterruns[i]==run && filterevts[i]==event` (subRun ignored).

Port (minimal — no code change):
- Its only *active* includes are art/canvas/fhicl (the sbndcode/larsoft includes are
  commented out) → **no sbndcode dependency**, copied **verbatim** to
  `larwirecell/larwirecell/Modules/FilterEventID_module.cc`.
- Added to `larwirecell/larwirecell/Modules/CMakeLists.txt`:
  ```cmake
  cet_build_plugin(FilterEventID art::EDFilter
    LIBRARIES PRIVATE
    canvas::canvas
    fhiclcpp::fhiclcpp
    )
  ```
- Runnable fcl `samples/filter-nuecc-rse.fcl` = upstream fcl with
  `filterruns`/`filterevts` filled from the CSV; `RootOutput fileName:"%ifb_eventidfiltered.root"`,
  `SelectEvents:[ filter ]`.

### Build (MRB) — gotchas
- Build env: SL7 apptainer →
  ```
  source /cvmfs/sbnd.opensciencegrid.org/products/sbnd/setup_sbnd.sh
  cd /exp/sbnd/app/users/yuhw/larsoft-wct036/v10_14_02
  source localProducts_larsoft_v10_14_02_02_e26_prof/setup
  mrbsetenv
  cd "$MRB_BUILDDIR"; mrb b
  ```
- **Do NOT use raw `make`** in the build tree: `CC`/`CXX` are empty there, so the
  reconfigure resets `CMAKE_CXX_COMPILER` to bare `g++`, invalidates the cache, and
  fails (Range-v3 not found). `mrb b` sets the compiler correctly.
- `mrb b` currently **fails overall** on a *pre-existing, unrelated* error in the
  `qlmatch` subpackage: `WireCellIface/IDetectorVolumes.h: No such file`
  (local wire-cell-toolkit / larwirecell version skew). FilterEventID builds &
  links **first** (43–45%), so its `.so` is produced regardless of that later failure.

### Deploy — dual-tree hand-copy
Runtime loads larwirecell plugins from the **opt** install (via
`setup-local-opt.sh` → `CET_PLUGIN_PATH`), NOT from the MRB localProducts. So copy:
```
cp build_slf7.x86_64/larwirecell/slf7.x86_64.e26.prof/lib/liblarwirecell_Modules_FilterEventID_module.so \
   /exp/sbnd/app/users/yuhw/opt/larwirecell/v10_01_28/slf7.x86_64.e26.prof/lib/
```

### Run + result
```
# SL7 apptainer:
source setup-ap.sh
cd samples/filtered-reco1
lar -c ../filter-nuecc-rse.fcl -S ../lynn-nuecc-reco1-files.lst
```
Result: `TrigReport Events total = 2350  passed = 48  failed = 2302`, exit 0, no
errors. Output `samples/filtered-reco1/…_eventidfiltered.root` verified to contain
exactly **48** events (via `FileIndex`).

Note: `RootOutput` does not roll over per input file by default, so all 48 selected
events land in **one** output file (named from one input's `%ifb`). For one filtered
file per input reco1 file, add `fileProperties: { granularity: InputFile }`.

---

## Status / caveats
- **Nothing committed or pushed** — larwirecell edits (module + CMakeLists) and the
  copied `.so` are local, awaiting review.
- The `qlmatch` subpackage does not currently compile against the local toolkit
  (`IDetectorVolumes.h`); unrelated to this port but blocks a clean full `mrb b`.
- Run log: `samples/filter-nuecc-run.log`.
