# 3 – larwirecell qlmatch build failure: `WireCellIface/IDetectorVolumes.h` not found

**Status: RESOLVED + VERIFIED (2026-07-14).** Fixed by building against the local/opt
WireCell (`WIRECELL_FQ_DIR=/exp/.../opt` + opt on `CMAKE_PREFIX_PATH`) and forcing a
cmake **re-configure** so the stale cvmfs include path is replaced. See the canonical
recipe + the two gotchas (cmake-not-on-PATH regenerate; `make install` /usr/local
prefix trap) in `sbnd/docs/0-build-wct-larwirecell-sl7-sbnd.md`. `libWireCellQLMatch.so`
and `libWireCellAIML.so` rebuilt and installed to opt (verified to link opt WireCell).
The detail below explains the root cause.

## Symptom

`mrb b` of larwirecell fails while compiling the `qlmatch` subpackage:

```
larwirecell/qlmatch/QLMatching.h:7:10: fatal error:
    WireCellIface/IDetectorVolumes.h: No such file or directory
    7 | #include "WireCellIface/IDetectorVolumes.h"
gmake: *** [.../WireCellQLMatch.dir/QLMatching.cxx.o] Error 1
FATAL ERROR: stage build FAILED for MRB project larsoft v10_14_02_02 with code 2
```

## Root cause — WireCell header/library split

The header **does** exist, but not where the compiler looks:

| location | has `IDetectorVolumes.h`? |
|---|---|
| local toolkit src `wire-cell-toolkit/iface/inc/WireCellIface/` (branch `apply-pointcloud`) | **yes** |
| opt install `/exp/sbnd/app/users/yuhw/opt/include/WireCellIface/` | **yes** (mtime 2026-04-29) |
| cvmfs `wirecell/v0_32_1/…/include/WireCellIface/` | **NO** |

`IDetectorVolumes.h` is a **newer interface** added on the local `apply-pointcloud`
branch; it is absent from the older released `wirecell v0_32_1`.

The larwirecell build resolves WireCell inconsistently:

- **Compile (headers):** larwirecell's `ups/product_deps` pins `wirecell v0_32_1`,
  so `mrbsetenv` sets up the **cvmfs** wirecell v0_32_1 and its include dir lands on
  the qlmatch compile line (verified in `flags.make`):
  ```
  -isystem /cvmfs/larsoft.opensciencegrid.org/products/wirecell/v0_32_1/Linux64bit+3.10-2.17-e26-prof/include
  ```
  `/exp/.../opt/include` is **not** on the compile line → the new header is invisible.
- **Link (library):** the CMake cache points the WireCell lib at the **opt** build:
  ```
  WireCell_Iface_LIBRARY:FILEPATH=/exp/sbnd/app/users/yuhw/opt/lib/libWireCellIface.so
  ```

So qlmatch compiles against **cvmfs v0_32_1 headers** but links **opt's newer lib** —
and the source (`QLMatching.h`) needs an interface that only exists in the newer tree.

## Evidence trail
- Include used: `build_slf7.x86_64/larwirecell/larwirecell/qlmatch/CMakeFiles/WireCellQLMatch.dir/flags.make`
  → `CXX_INCLUDES` contains `wirecell/v0_32_1/.../include`, zero refs to `opt/include`.
- `ls …/wirecell/v0_32_1/…/include/WireCellIface/IDetectorVolumes.h` → No such file.
- `larwirecell/ups/product_deps` → `wirecell  v0_32_1`.
- `QLMatching.h:7` includes it; `QLMatching.h:71` uses `IDetectorVolumes::pointer m_dv;`.

## Impact
- Full `mrb b` / `mrb i` of larwirecell cannot complete → the qlmatch runtime libs are
  not rebuilt/installed from this tree. (The libs already in `opt` from the last good
  build on 2026-06-23 still work at runtime; only rebuilding is blocked.)
- Unrelated to and not blocking `FilterEventID` (built + deployed independently).

## Fix directions (NOT applied — for review)
The build must take WireCell **headers** from the same tree as the linked lib (the
`apply-pointcloud`/opt WireCell that has `IDetectorVolumes.h`), instead of cvmfs
`v0_32_1`. Options, roughly in order of cleanliness:

1. **Repoint the WireCell ups product** larwirecell depends on to a local build that
   matches opt (i.e. an `apply-pointcloud` wirecell ups product with the new headers),
   updating `larwirecell/ups/product_deps` (`wirecell v0_32_1` → local version) so
   `mrbsetenv` sets up the matching `WIRECELL_INC`.
2. **Force find_package(WireCell) to opt** at configure time — prepend
   `/exp/sbnd/app/users/yuhw/opt` to `CMAKE_PREFIX_PATH` (and/or set `WireCell_DIR`)
   before `mrbsetenv`/`mrb b` so both headers and libs resolve from opt. Then a clean
   reconfigure (delete the stale `WireCell_*` cache entries) so the `-isystem` include
   flips to `opt/include`.
3. **Bump the pinned wirecell version** to a released tag that already contains
   `IDetectorVolumes.h` (only if such a release exists and is ABI-compatible).

Whichever is chosen, do a clean reconfigure afterward (`flags.make` currently caches
the cvmfs include path) and confirm the qlmatch `-isystem` points at the tree that
has `IDetectorVolumes.h`.
