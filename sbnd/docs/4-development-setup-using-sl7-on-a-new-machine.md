# Development setup: SBND WCT/larwirecell on SL7 (new machine)

For Avinay — how our imaging/clustering/QL-matching + nugraph development is
laid out, which branches we track, and how to build & run it inside the SL7
apptainer on a fresh machine.

## 1. Repos & branches

| Repo | Our checkout (yuhw) | Branch | Remote of that branch |
|---|---|---|---|
| **wire-cell-toolkit** (WCT) | `/exp/sbnd/app/users/yuhw/wire-cell-toolkit` | `ap-yuhw` | `fork` = `git@github.com:HaiwangYu/wire-cell-toolkit.git` (branched from `apply-pointcloud` @ `0442bc27`; `origin` = `WireCell/wire-cell-toolkit`) |
| **larwirecell** (MRB tree) | `/exp/sbnd/app/users/yuhw/larsoft-wct036/v10_14_02/srcs/larwirecell` | `dev-v10_14_02_02` | `origin` = `git@github.com:HaiwangYu/larwirecell.git` |
| **wcp-porting-validation** | `/exp/sbnd/app/users/yuhw/wcp-porting-img` | `main` | `git@github.com:WireCell/wcp-porting-validation.git` |

Notes:
- The **`sbnd/`** subdir of `wcp-porting-validation` holds the fcls, jsonnet
  (`wcls-img-clus-matching-xin[-data].fcl`, entry jsonnet), the setup scripts,
  `sbnd_xin/` (the standalone Q/L + PR chain), and these `docs/`.
- Use **only** the MRB larwirecell tree above. Do **not** use
  `/exp/sbnd/app/users/yuhw/larwirecell` (that is a source-only dev tree).
- Recent feature commits: WCT `ap-yuhw` has the `opflash_time` column on
  `clustering-global`; larwirecell `dev-v10_14_02_02` has the `wclsTensorSetLabeler`
  (tagger_lm / sed sets / mc-tree nu info / the ctpc-nugraph fix).

## 2. SL7 container

Everything runs inside the Fermilab SL7 apptainer. The machine only needs the
standard CVMFS + shared mounts; nothing is installed on the host.

```bash
/cvmfs/oasis.opensciencegrid.org/mis/apptainer/current/bin/apptainer exec \
  -B /cvmfs,/exp,/nashome,/pnfs \
  /cvmfs/singularity.opensciencegrid.org/fermilab/fnal-dev-sl7:latest bash -c '
    source ~/.bashrc                 # non-interactive shells need this first
    source <setup script>            # see below
    cd <workdir>
    <lar / wire-cell / wcb ...>
  '
```

- **Requirements on the new machine:** `/cvmfs`, `/exp`, `/nashome`, `/pnfs`
  mounted (any SBND gpvm/build node has these). Our code + the `opt` install
  live on `/exp`, which is shared, so if you can read yuhw's `/exp` paths you
  can run directly; to develop your own copy, clone the branches into your own
  `/exp/sbnd/app/users/<you>/` area and rebuild (Section 4).
- `source ~/.bashrc` is required in non-interactive shells (sets up UPS etc.).

### Setup scripts (in `sbnd/`)
- `setup-local-opt.sh` — legacy `opt` env; **sbndcode cfg wins**. Use for **sim**
  and the old matching chain.
- `setup-ap.sh` — the AP imaging/matching env: `setup-local-opt.sh` **+ prepend
  the toolkit `cfg`** (so toolkit img/clus/qlmatching/simparams win) + `sbnd_xin`
  + `wire-cell-data`. Use for the **1-step img→clus→QL→taggers→labeler chain**.

Both scripts point at yuhw's `opt` (`/exp/sbnd/app/users/yuhw/opt`) and paths —
copy and edit them to your own `opt`/checkout paths.

## 3. The `opt` install layout
- `/exp/sbnd/app/users/yuhw/opt` — WCT libs/headers/cfg (`lib/`, `include/`,
  `share/wirecell`).
- `/exp/sbnd/app/users/yuhw/opt/larwirecell/v10_01_28/slf7.x86_64.e26.prof/lib`
  — the hand-copied larwirecell `.so` (`libWireCellLarsoft/AIML/QLMatch.so`,
  `liblarwirecell_Modules_*`).

## 4. Building

### WCT
```bash
# inside SL7, after: source ~/.bashrc; source sbnd/setup-ap.sh
cd <wire-cell-toolkit>
./wcb -j16 --notests install     # installs into $WIRECELL_FQ_DIR (= opt)
```
- Externals used: `spdlog v1_14_1`, `fmt v11_0_2` (CVMFS), no spng.
- **Run the build in the FOREGROUND with a long timeout.** Long background
  jobs on our setup got killed spuriously (exit 144) while the real build kept
  running orphaned — foreground avoids that. Full build ~5 min.
- Editing `util/inc/WireCellUtil/Bee.h` (or any util header) recompiles almost
  everything and changes ABI — then you must rebuild larwirecell too. Prefer
  keeping changes in `clus`/`root` when possible.

### larwirecell (MRB)
```bash
# inside SL7, after: source ~/.bashrc; source sbnd/setup-local-opt.sh
source /exp/sbnd/app/users/yuhw/larsoft-wct036/v10_14_02/localProducts_larsoft_v10_14_02_02_e26_prof/setup
mrbsetenv
# CRITICAL: force WCT to opt AFTER mrbsetenv (mrb prepends CVMFS wirecell/v0_32_1)
export CMAKE_PREFIX_PATH=/exp/sbnd/app/users/yuhw/opt:$CMAKE_PREFIX_PATH
export WIRECELL_INC=/exp/sbnd/app/users/yuhw/opt/include
export WIRECELL_LIB=/exp/sbnd/app/users/yuhw/opt/lib
export WIRECELL_FQ_DIR=/exp/sbnd/app/users/yuhw/opt
cd $MRB_BUILDDIR/larwirecell
make -j16                        # (or `make -j16 WireCellAIML` for just the labeler)
# hand-copy the freshly built .so into opt:
cp $MRB_BUILDDIR/larwirecell/lib/libWireCell*.so \
   /exp/sbnd/app/users/yuhw/opt/larwirecell/v10_01_28/slf7.x86_64.e26.prof/lib/
```

**larwirecell build landmines (all real, all bit us):**
1. **`Aux::taginfo` ABI (taginfo 1-arg vs 2-arg):** `mrbsetenv` sets up the
   CVMFS `wirecell/v0_32_1` UPS product and prepends it to `CMAKE_PREFIX_PATH`,
   so without the opt-first override above the code compiles against the wrong
   (old) WCT headers → "undefined symbol …Aux…taginfo…" at load. Set
   `CMAKE_PREFIX_PATH`/`WIRECELL_INC/LIB/FQ_DIR` to `opt` **after** `mrbsetenv`.
2. **Stale objects / staging:** for a clean rebuild, delete all `*.cxx.o` (not
   just relink) and the stale `build.../larwirecell/slf7.x86_64.e26.prof/lib/*.so`,
   then `make`.
3. **DT_RPATH:** the hand-copied `.so` may carry an RPATH that puts CVMFS
   `wirecell/v0_32_1/lib` first (shadowing opt). Fix after copy:
   `patchelf --force-rpath --set-rpath "<opt>/lib:$(patchelf --print-rpath <so>)" <so>`
   on `libWireCellLarsoft.so`, `libWireCellAIML.so`, `libWireCellQLMatch.so`.
4. **`miniz.h`:** `WireCellUtil/Bee.h` → `custard_boost.hpp` → `miniz.h`, which
   `wcb install` does NOT install. Hand-copy
   `<wct>/util/inc/WireCellUtil/custard/miniz.h` into
   `<opt>/include/WireCellUtil/custard/` after each fresh WCT install.

## 5. Running the chain

```bash
# inside SL7, after: source ~/.bashrc; source sbnd/setup-ap.sh
export FHICL_FILE_PATH=<...>/wcp-porting-img/sbnd:$FHICL_FILE_PATH
cd <run dir>
# MC 1-step:
lar -n 1 --nskip K -c wcls-img-clus-matching-xin.fcl      -s <reco1_mc.root>   --no-output
# DATA 1-step (reality=data; needs a *_frameshift.root input for Gen2 data):
lar -n 1 --nskip K -c wcls-img-clus-matching-xin-data.fcl -s <reco1_data.root> --no-output
```
- **Run per event** (`-n 1 --nskip K`, wrap in `timeout`). Inline multi-event
  batches SIGSEGV in the tagger/steiner patrec on some events; per-event
  isolation contains a bad event to itself.
- Output per event: `mabc.zip` (BEE point sets: clustering-global +
  opflash_time, tagger_fc/stm/tgm/lm, and for MC the truth/sed/mc layers) and
  `nugraph.h5` (pynuml graph; ctpc sp-sp edges). Drop `trash-all-apa.tar.gz`,
  `*.db`, `tf-default.root`, `mabc-pr.zip`.
- BEE upload: `BROWSER=echo bash sbnd/sbnd_xin/upload-to-bee.sh <file.zip>`.
- nugraph.h5 → BEE point view: `sbnd/TensorSetLabeler/h5_sp_to_bee.py <nugraph.h5> <out.zip>`.

## 6. More detail (existing docs in this dir)
- `docs/0-build-wct-larwirecell-sl7-sbnd.md` — full build recipe + landmines.
- `docs/1-run-tests-sl7-local-builds-sbnd.md` — how to run the chain, toggles,
  log greps, validation, BEE upload.
- Base SL7 how-to: `/exp/sbnd/app/users/yuhw/claude-utilities/wct-in-sl7.md`
  (and `in-gpvm-sl7.sh`, a wrapper that runs a command inside the apptainer).
