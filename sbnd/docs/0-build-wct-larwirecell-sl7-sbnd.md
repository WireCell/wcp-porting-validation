# Build WCT + larwirecell and install into /exp/sbnd/app/users/yuhw/opt (SL7, SBND)

The canonical build/install recipe for this working area.  Everything is
SBND-specific (sbndcode v10_14_02_03, larsoft v10_14_02, e26:prof, SL7);
for another experiment (e.g. DUNE) the setup scripts, MRB area, product
versions and the cfg/WIRECELL_PATH layering all need adapting.

## Layout

| what | where |
|---|---|
| wire-cell-toolkit source | `/exp/sbnd/app/users/yuhw/wire-cell-toolkit` |
| larwirecell source (MRB tree — the ONLY one to build) | `/exp/sbnd/app/users/yuhw/larsoft-wct036/v10_14_02/srcs/larwirecell` |
| MRB build dir | `/exp/sbnd/app/users/yuhw/larsoft-wct036/v10_14_02/build_slf7.x86_64` |
| local install (runtime loads THIS) | `/exp/sbnd/app/users/yuhw/opt` (WCT libs in `opt/lib`, cfg in `opt/share/wirecell`, larwirecell in `opt/larwirecell/v10_01_28/slf7.x86_64.e26.prof/lib`) |
| SL7 wrapper | `/exp/sbnd/app/users/yuhw/claude-utilities/in-gpvm-sl7.sh` (its baked-in `wcdev-sbnd/setup.sh` does NOT exist — always source a setup script yourself inside `bash -c`) |
| build env setup | `wcp-porting-img/sbnd/setup-local-opt.sh` |
| run env setup (AP chain) | `wcp-porting-img/sbnd/setup-ap.sh` |

Everything builds and runs INSIDE the SL7 apptainer; non-interactive shells
must source `~/.bashrc` first (defines `path-prepend` etc.).

## 1. Build + install wire-cell-toolkit

```bash
/exp/sbnd/app/users/yuhw/claude-utilities/in-gpvm-sl7.sh bash -c '
source /nashome/y/yuhw/.bashrc >/dev/null 2>&1
source /exp/sbnd/app/users/yuhw/wcp-porting-img/sbnd/setup-local-opt.sh >/dev/null 2>&1
cd /exp/sbnd/app/users/yuhw/wire-cell-toolkit
./wcb -j16 --notests install
'
```

- The `build/` cache in the source tree already carries the configure
  (prefix = `/exp/sbnd/app/users/yuhw/opt`); a from-scratch configure would
  be `./wcb configure --prefix=/exp/sbnd/app/users/yuhw/opt ...` (see
  `build/config.log` for the full original option set).
- **Always `install`**, never build-only: a stale `opt` gives
  undefined-symbol / vtable errors at plugin load time.
- `install` also copies `cfg/` → `opt/share/wirecell`.  Note the
  WIRECELL_PATH layering: `setup-ap.sh` prepends the toolkit SOURCE `cfg/`
  (so source-tree jsonnet edits take effect without reinstalling);
  `setup-local-opt.sh` puts `opt/share/wirecell` ahead of the sbndcode cfg.
- Timing: full rebuild ≈ 5 min, incremental ≈ 2–3 min (`-j16` is fine).
- Verify a new component landed:
  `nm -DC /exp/sbnd/app/users/yuhw/opt/lib/libWireCellClus.so | grep <Name>`.
- Unit tests: `./wcb -j16` (builds tests too), then run e.g.
  `./build/clus/wcdoctest-clus`.

## 2. Build larwirecell against the local WCT (MRB)

```bash
/exp/sbnd/app/users/yuhw/claude-utilities/in-gpvm-sl7.sh bash -c '
source /nashome/y/yuhw/.bashrc >/dev/null 2>&1
source /exp/sbnd/app/users/yuhw/wcp-porting-img/sbnd/setup-local-opt.sh >/dev/null 2>&1
# point any cmake re-run at the LOCAL WCT (else FindWireCell falls back to
# the cvmfs WCT product, which lacks the new headers):
export WIRECELL_FQ_DIR=/exp/sbnd/app/users/yuhw/opt
export CMAKE_PREFIX_PATH=/exp/sbnd/app/users/yuhw/opt:$CMAKE_PREFIX_PATH
source /exp/sbnd/app/users/yuhw/larsoft-wct036/v10_14_02/localProducts_larsoft_v10_14_02_02_e26_prof/setup >/dev/null 2>&1
mrbsetenv >/dev/null 2>&1
cd $MRB_BUILDDIR/larwirecell
make -j16 install
'
```

- `mrbsetenv` (NOT just `mrbslp`) is required — plain mrbslp hits a larevt
  version conflict and does not set the build env.
- `make` re-runs cmake automatically when a CMakeLists.txt changed; that is
  when `WIRECELL_FQ_DIR`/`CMAKE_PREFIX_PATH` matter.  Avoid a bare
  `cmake $MRB_SOURCE/larwirecell` (it loses `CMAKE_INSTALL_PREFIX`).
- **Check the build exit status before installing/copying** — a failed
  build followed by a blind copy runs STALE code silently.

## 3. Hand-copy the larwirecell .so into opt (required!)

`make install` lands in `$MRB_INSTALL` (localProducts...), but the runtime
(`setup-local-opt.sh` → `CET_PLUGIN_PATH`/`LD_LIBRARY_PATH`) loads
`/exp/sbnd/app/users/yuhw/opt/larwirecell/...` — separate inodes, so:

```bash
cp $MRB_INSTALL/larwirecell/v10_01_28/slf7.x86_64.e26.prof/lib/libWireCellAIML.so \
   /exp/sbnd/app/users/yuhw/opt/larwirecell/v10_01_28/slf7.x86_64.e26.prof/lib/
# (copy whichever lib*.so you rebuilt: libWireCellLarsoft.so,
#  libWireCellQLMatch.so, libWireCellAIML.so, ...)
```

Verify: `nm -DC .../opt/larwirecell/.../lib/libWireCellAIML.so | grep <Name>`.

## Known landmines

- **miniz.h**: `WireCellUtil/Bee.h` → `custard_boost.hpp` → `miniz.h`,
  which `wcb install` does NOT install.  One-time (per fresh opt) fix:
  `cp wire-cell-toolkit/util/inc/WireCellUtil/custard/miniz.h
      /exp/sbnd/app/users/yuhw/opt/include/WireCellUtil/custard/`
- **-Werror traps** (larwirecell): don't bind
  `face->sensitive().bounds()` to a reference (dangling temporary);
  constructor init lists must follow declaration order (`-Werror=reorder`).
- The wcb-built WCT and the cvmfs WCT product must not mix: if larwirecell
  compiles against cvmfs headers you get missing-header errors (e.g.
  `IDetectorVolumes.h`) — that means `WIRECELL_FQ_DIR` wasn't set when
  cmake re-ran.
- Run-time env: `setup-local-opt.sh` for sim + legacy chain,
  `setup-ap.sh` for the AP/Xin imaging-clustering-matching chain.  Both
  load the SAME `opt` install.

## Caveats when the WireCell fix "doesn't take" (verified 2026-07-14)

The `WIRECELL_FQ_DIR` / `CMAKE_PREFIX_PATH` fix above only helps **while cmake
actually re-configures**.  If you set the env and just `make`, and no
`CMakeLists.txt` changed since the last (bad) configure, `make` does **not**
re-run cmake — it reuses the stale `flags.make`/`link.txt` (still pointing at
cvmfs `wirecell/v0_32_1`) and fails with the SAME `IDetectorVolumes.h` /
`IBeeSink.h` error.  Symptoms & how to force a real reconfigure:

- **cmake is NOT on PATH** in this SL7+mrb env, so a manual `cmake .` dies with
  `command not found` (silently, if you don't check).  Use the cached full path:
  `/cvmfs/larsoft.opensciencegrid.org/products/cmake/v3_27_4/Linux64bit+3.10-2.17/bin/cmake .`
  (it's `CMAKE_COMMAND` in `$MRB_BUILDDIR/larwirecell/CMakeCache.txt`), run from
  `$MRB_BUILDDIR/larwirecell` with the opt env set.  Or `touch` any
  `CMakeLists.txt` so plain `make` triggers the cached cmake.
- Confirm it worked: the configure log prints
  `FindWireCell: WireCell_INCLUDE_DIR = /exp/.../opt/include`, and
  `larwirecell/qlmatch/CMakeFiles/WireCellQLMatch.dir/flags.make` `CXX_INCLUDES`
  now shows `.../opt/include` with NO `wirecell/v0_32_1`.
- Stale-cache tell: `WireCell_*_LIBRARY` entries in the cache can be a MIX of opt
  and cvmfs from earlier runs.  If a clean reconfigure isn't flipping them, unset
  them and re-resolve: `cmake -U 'WIRE-CELL' -U 'WireCell_INCLUDE_DIR' -U 'WireCell_*_LIBRARY' .`

## `make install` prefix trap (verified 2026-07-14)

In this build tree `CMAKE_INSTALL_PREFIX` is `/usr/local` (read-only), so
`make -j16 install` dies with `Read-only file system` AFTER a successful compile
— the libs build fine, only the install step fails.  Consequences:

- The freshly-built libs land in **`$MRB_BUILDDIR/larwirecell/lib/`** (the CMake
  `LIBRARY_OUTPUT_DIRECTORY`), NOT in `$MRB_BUILDDIR/larwirecell/slf7.x86_64.e26.prof/lib/`
  (the cetmodules fq/EXEC_PREFIX path, which only updates on a successful
  `make install`).  Hand-copy from `.../larwirecell/lib/`.
- The two output copies differ only by debug info: `larwirecell/lib/` is smaller
  (no `debug_info`), the fq-path copy is larger (`with debug_info`).  Both are
  valid, loadable plugins.
- Deploy step 3 therefore becomes (when `make install` can't be used):
  ```bash
  cp $MRB_BUILDDIR/larwirecell/lib/libWireCellQLMatch.so \
     $MRB_BUILDDIR/larwirecell/lib/libWireCellAIML.so \
     /exp/sbnd/app/users/yuhw/opt/larwirecell/v10_01_28/slf7.x86_64.e26.prof/lib/
  ```
  Verify the installed lib links opt (not cvmfs) WireCell:
  `ldd .../opt/larwirecell/.../lib/libWireCellQLMatch.so | grep libWireCell`
  → every `libWireCell*.so` must resolve under `/exp/.../opt/lib`.
- (Proper fix, if you want `make install` back: reconfigure with
  `-DCMAKE_INSTALL_PREFIX=$MRB_INSTALL`.)

## More context

- `sbnd/docs/claude-session-20260623-20260707.md` — run recipes (MC/data
  chains, wcsonnet checks, BEE upload).
- larwirecell `aiml/docs/TensorSetLabeler-notes.md` — component-level
  pitfalls (this doc's build section distilled from the same sessions).
