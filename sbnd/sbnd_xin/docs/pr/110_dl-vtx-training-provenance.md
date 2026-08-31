# doc pr/110 — Provenance of the uBooNE SCN vertex net: was it trained on a production *without* the exclusion fit? (2026-08-22)

**Status:** git/GitHub archaeology only. **No code, no config, no behavior change — no A/B gate applies.**

**Owner question (2026-08-22):** "In toolkit, when exclude fit is off, the DL vertex has a
significantly better performance, this may be explained by a better match with the trained model.
We also know that with the exclude fit, the pattern recognition near vertex looks better, but the
DL Vtx performs worse. Because of this, I wonder whether it is possible that the model MicroBooNE
trained by Haiwang was on images without the exclusion fit? 1. with the prototype code, check the
history of when I invent or integrated the exclusion fit into the uboone production? 2. check a bit
the history of Haiwang's training directory (in github) … Basically, if he might be using a version
of prototype code without exclusion fit."

**Answer, bracketed:**

* The premise that the training cloud is *exclusion-sensitive* is **correct and now proven from
  the code** (§3): Haiwang's loader reads `T_rec_charge_blob`, which is the **fitted trajectory
  point cloud with `dQ·0.1 − 1000`** — the very same construction the DL inference cloud is built
  from, on both the prototype and the toolkit. So the hypothesis is well posed, not a category error.
* But a **fully** exclusion-free training sample is **very unlikely** (§4). The tree the loader
  needs (`T_rec_charge_blob`) first appears anywhere in `wire-cell-pid` on **2020-03-17**; the
  exclusion fit's first live call site landed **2020-03-24**, seven days later, and saturated at 29
  call sites by **2020-05-11** — a full month before Haiwang's first training commit (2020-06-10).
  A no-exclusion sample requires production inside that 7-day window, then ~3 months on the shelf.
* What **remains live**, and is not settled here, is a *partial*-rollout training distribution: a
  sample produced late-March/early-April 2020 would have carried only **1–8 of the eventual 29**
  exclusion call sites. That is a genuinely different charge cloud from today's.
* What is **certain** is that the net never saw the **toolkit's** exclusion fit. pr/108 §5 and
  pr/109 show the toolkit's exclusion is *not functionally identical* to WCP's on SBND, and pr/109
  shows it is *worse* at describing the measured 2-D charge near the SBND vertex on 6/6 events.
  **This is the better-evidenced explanation of the DL-off gain** and is given equal billing in §5.

Related: pr/106 §9 (`dl_vtx_cloud_no_exclusion`, nueCC 35 → 41 vertex, but nue-selected 35 → 32),
pr/107 (`dqdx_fit_keep_all_points`), pr/108 (exclusion-fit parity WCP vs WCT), pr/109 (2-D charge
residual, exclusion ON vs OFF).

---

## 0. Repro block

All read-only. `gh` must be authenticated (`gh auth status`).

```bash
PROTO=/nfs/data/1/xqian/prototype-dev/wire-cell     # BNLIF/wire-cell superproject
PID=$PROTO/pid                                      # BNLIF/wire-cell-pid

# --- Q1: birth of the exclusion fit -------------------------------------------------
git -C $PID log --all --format='%h %ad %an %s' --date=short -S 'flag_exclusion' -- .
git -C $PID show --stat c16dce5 | head -12
git -C $PID log -1 --format='%h %ai %s' e659ad3

# rollout curve: number of call sites passing flag_exclusion=true, per commit
prev=""; for c in $(git -C $PID log --format='%h' --reverse --since=2020-03-01 --until=2020-07-15 -- src/); do
  n=$(for f in $(git -C $PID ls-tree -r --name-only $c src/ | grep -E '\.h$'); do git -C $PID show $c:$f 2>/dev/null; done \
        | grep -c "microsecond, true, true, true")
  [ "$n" != "$prev" ] && { echo "$(git -C $PID log -1 --format='%ad' --date=short $c) $c callers=$n"; prev=$n; }
done

# release tags -> pid submodule SHA -> exclusion caller count
for t in $(git -C $PROTO tag | grep -E '^v00_1') ; do
  s=$(git -C $PROTO ls-tree $t pid | awk '{print $3}')
  printf "%-14s %s pid=%s %s callers=%s\n" $t "$(git -C $PROTO log -1 --format=%ad --date=short $t)" \
    ${s:0:8} "$(git -C $PID log -1 --format=%ad --date=short $s)" \
    "$(git -C $PID show $s:src/NeutrinoID_proto_vertex.h 2>/dev/null | grep -c 'microsecond, true, true, true')"
done
# the v00_12_xx / v00_13_00..03 releases are BRANCHES, not tags -- same loop over
#   origin/v00_12_00 origin/v00_12_01 origin/v00_13_00 origin/v00_13_01 origin/v00_13_02 origin/v00_13_03

# --- Q2: Haiwang's training repo ---------------------------------------------------
gh api users/HaiwangYu/repos --paginate --jq '.[]|[.name,.created_at[0:10],.pushed_at[0:10]]|@tsv' | sort -k2
gh api repos/HaiwangYu/uboone-dl-vtx --jq '[.created_at,.pushed_at]|@tsv'
gh api 'repos/HaiwangYu/uboone-dl-vtx/commits?per_page=100' --paginate \
  --jq '.[]|[.sha[0:8],.commit.author.date[0:10],(.commit.message|split("\n")[0])]|@tsv'
gh api repos/HaiwangYu/uboone-dl-vtx/git/trees/master?recursive=1 --jq '.tree[]|[.type,.path]|@tsv'
gh api 'repos/HaiwangYu/uboone-dl-vtx/contents/util/loader.py?ref=master' --jq .content | base64 -d
gh api 'repos/HaiwangYu/uboone-dl-vtx/contents/train1.py?ref=master'      --jq .content | base64 -d
gh api 'repos/HaiwangYu/uboone-dl-vtx/contents/.gitignore?ref=master'     --jq .content | base64 -d
# which checkpoint hybrid_vtx.py points at, per commit (CP19 -> CP24 on 2020-08-06)
for s in 7c2403cc d0036b97 84d1eeae 8108f05a; do
  printf '%s %s ' $s "$(gh api repos/HaiwangYu/uboone-dl-vtx/commits/$s --jq '.commit.author.date[0:10]')"
  gh api "repos/HaiwangYu/uboone-dl-vtx/contents/hybrid_vtx.py?ref=$s" --jq .content | base64 -d | grep "^ *model_path = 't48k"
done
gh api repos/BNLIF/wire-cell-pydata/git/trees/master?recursive=1 --jq '.tree[]|[.type,.path,.size]|@tsv'
gh api repos/BNLIF/wire-cell-pydata/commits --jq '.[]|[.sha[0:8],.commit.author.date[0:10],(.commit.message|split("\n")[0])]|@tsv'
gh api 'repos/HaiwangYu/uboone-prod/contents/Reco1.5-Reco2/grid/reco1.5-reco2_WCPport.xml' \
  --jq .content | base64 -d | grep -iE 'release|WCP_v'

# --- the dating constraint ----------------------------------------------------------
git -C $PID log --all --format='%h %ad %s' --date=short -S 'T_rec_charge_blob' -- apps/
# which apps can produce a file the loader can open (needs T_vtx type+flag_main)
cd $PROTO && for f in $(grep -rl 'T_rec_charge_blob' --include=*.cxx . | grep -v no_support); do
  printf "%-64s T_vtx=%s flag_main=%s fill_skeleton_info=%s\n" $f \
    $(grep -c '"T_vtx"' $f) $(grep -c flag_main $f) $(grep -c fill_skeleton_info $f); done
git -C $PID show b0d46f5:apps/wire-cell-prod-nue.cxx | grep -nE 'T_rec_charge_blob|"T_vtx"'
git -C $PID show b0d46f5:apps/wire-cell-prod-nue.cxx | sed -n '1328,1346p'   # T_vtx branches
for t in T_rec_charge_blob T_vtx flag_main; do
  echo -n "$t in v00_13_03: "; git -C $PID show 7921bfb9:apps/wire-cell-prod-nue.cxx | grep -c "$t"; done

# --- checkpoint format --------------------------------------------------------------
cd /nfs/data/1/xqian/toolkit-dev/wire-cell-data/uboone/scn_vtx
file t48k-m16-l5-lr5d-res0.5-CP24.pth        # -> "data" (NOT a zip => legacy torch format)
unzip -l t48k-m16-l5-lr5d-res0.5-CP24.pth    # -> "End-of-central-directory signature not found"
head -c 4000 t48k-m16-l5-lr5d-res0.5-CP24.pth | strings | head -20   # bare state_dict, no metadata
git -C /nfs/data/1/xqian/toolkit-dev/wire-cell-data log --format='%h %ad %s' --date=short -- uboone/scn_vtx
```

---

## 1. Q1 — when the exclusion fit was invented and integrated into uBooNE production

`flag_exclusion` is the 8th argument of `WCPPID::PR3DCluster::do_multi_tracking`
(`pid/inc/WCPPID/PR3DCluster.h:182`, default `false`), threaded into
`organize_segments_path_2nd_order`/`update_association`
(`pid/src/PR3DCluster_multi_track_fitting.h:970-1096`).

### 1.1 Birth and rollout (repo `BNLIF/wire-cell-pid`, all commits by Xin Qian)

| when | commit | what |
|---|---|---|
| **2020-03-24 19:06:05 -0400** | `c16dce5` "improve" | **the parameter is born.** `flag_exclusion = false` added to `do_multi_tracking` and to the internal fitting signature. Nobody passes `true` yet ⇒ still a no-op. |
| **2020-03-24 19:22:44 -0400** | `e659ad3` "improve" | **first live call site**, inside `NeutrinoID::find_other_segments`. |
| 2020-03-24 22:05 | `baa35ba` | 2 call sites |
| 2020-03-25 … 03-29 | `a0cbe70` → `a41dccb` | 3 → 8 |
| 2020-04-16 … 04-30 | `b7e5976` → `a4669ea` | 9 → 21 |
| 2020-05-01 … **2020-05-11** | `233b7c2` → **`d2e3587`** | 23 → **29 (saturated)** |

Full curve (one row per change), from the Repro block:

```
2020-03-01 3081b6c callers=0     2020-04-18 9f5d8ce callers=15
2020-03-24 e659ad3 callers=1     2020-04-19 73ee177 callers=14
2020-03-24 baa35ba callers=2     2020-04-19 19be70b callers=15
2020-03-25 a0cbe70 callers=3     2020-04-21 f15a8cf callers=17
2020-03-27 3598094 callers=4     2020-04-22 89c63df callers=19
2020-03-27 c68d9fe callers=5     2020-04-25 dddd584 callers=20
2020-03-28 b31b90c callers=6     2020-04-30 a4669ea callers=21
2020-03-28 6e4271e callers=7     2020-05-01 233b7c2 callers=23
2020-03-29 a41dccb callers=8     2020-05-01 41cd034 callers=24
2020-04-16 b7e5976 callers=9     2020-05-04 9bbd09a callers=25
2020-04-17 f3faf69 callers=10    2020-05-04 b5513f9 callers=26
2020-04-17 786dc0d callers=13    2020-05-06 9312b10 callers=27
2020-04-17 ff9e705 callers=14    2020-05-11 bc4cda4 callers=28
                                 2020-05-11 d2e3587 callers=29
```

Two later, unrelated touches of the token: `ab00269` (2024-12-30, "add more code to trajectory
fitting") and this year's env-gated debug hooks `d249b8e`/`8bd7b2b` (2026-08-21, doc pr/108).

### 1.2 Which uBooNE *production release* first carries it

The superproject `BNLIF/wire-cell` pins a `pid` SHA per release. Counting exclusion call sites in
`NeutrinoID_proto_vertex.h` at each pinned SHA:

| release | tag/branch date | pinned pid | pid date | exclusion callers |
|---|---|---|---|---|
| `v00_11_00` | 2019-09-13 | `99b49404` | 2019-09-13 | 0 |
| `v00_12_00` (branch) | 2019-11-07 | `013454e4` | 2019-11-05 | 0 |
| `v00_12_01` (branch) | 2019-11-19 | `016bd5ce` | 2019-11-19 | 0 |
| `v00_13_00` (branch) | 2019-11-20 | `69fa4db1` | 2019-11-20 | 0 |
| `v00_13_01` (branch) | 2019-12-04 | `131b22f0` | 2019-12-04 | 0 |
| `v00_13_02` (branch) | 2019-12-06 | `ee4c4a35` | 2019-12-06 | 0 |
| **`v00_13_03`** (branch) | **2020-02-10** | `7921bfb9` | 2020-01-22 | **0 — last release without it** |
| **`v00_13_04`** | **2020-06-17** | `7d45b632` | 2020-06-17 | **16 — first release with it** |
| `v00_13_05` … `v00_18_01` | 2020-06-30 … 2025-03-20 | — | — | 16 (every one) |

> **Answer to Q1.** The exclusion fit was invented on **2020-03-24** (`c16dce5` + `e659ad3`),
> rolled out over seven weeks to 29 call sites by **2020-05-11**, and first shipped in a tagged
> uBooNE production release as **`v00_13_04` (2020-06-17)**. Every release from then on has it.

MicroBooNE grid production ships WCP as a tarball of a tagged release — e.g.
`HaiwangYu/uboone-prod/Reco1.5-Reco2/grid/reco1.5-reco2_WCPport.xml` (Oct 2020) declares
`<!ENTITY release "WCP-001404">` and `--tar_file_name=…/WCP_v00_14_04.tar`. Note this convention
for §4: it is the *tagged* releases that get run at scale on the grid.

---

## 2. Q2 — Haiwang's training directory and the weights we actually ship

We run `dl_weights='uboone/scn_vtx/t48k-m16-l5-lr5d-res0.5-CP24.pth'`
(`cfg/pgrapher/experiment/sbnd/clus.jsonnet:728`). The name decodes as
*t48k* = training-sample size, *m16* = 16 UNet features, *l5* = 5 levels, *lr5d* = lr decay 5e-2,
*res0.5* = 0.5 cm voxels, *CP24* = checkpoint 24.

### 2.1 Provenance chain of that exact file

| date | where | commit |
|---|---|---|
| 2020-06-10 | **`HaiwangYu/uboone-dl-vtx` created** — the training directory | `dbfc7973` "Initial commit", `d7cf74e2` "init: forward working" |
| 2020-06-18 | `DeepVtx` model defined | `1e39595d` "init DeepVtx", `8d55d411` "model working" |
| 2020-06-22/23 | first end-to-end training | `12f37322` "test train predict working", `5e8dafca` "gpu worked" |
| 2020-06-25 | first sizeable run | `3480f1c9` "save 5k training 0625" |
| 2020-07-06 | 16k-event model | `84bcd8a8` "**t16k**" |
| 2020-07-09 | `257301ed` "v1.0" | |
| 2020-07-25 | the **`t48k/m16-l5-lr5d-res0.5`** series is in use — `hybrid_vtx.py` runs `CP19` | `7c2403cc` "save for collaboration meeting" |
| 2020-07-26 | still `CP19` | `d0036b97` |
| 2020-07-29 | `BNLIF/wire-cell-pyutil` created — the C++↔Python bridge | `c8f6dcf` "init readme" |
| 2020-08-01 | `pid/src/NeutrinoID_DL.h` added — DL vertex hooked into WCP | `89e8298` |
| 2020-08-05 | `BNLIF/wire-cell-pyinf` created (`SCN_Vertex.py`, `SCN/DeepVtx.py`) | `9383c37`, `31b15e3` "working with own state dict" |
| **2020-08-06** | **`BNLIF/wire-cell-pydata` created and the weights committed** | `a42aa2ba` init, **`059dbb19` "add current model"**, `6cd0623b` "nue-cc sample" |
| **2020-08-06** | **`hybrid_vtx.py` switches to `t48k/m16-l5-lr5d-res0.5/CP24.pth`** — the shipped checkpoint, same day it is published | `84d1eeae` "save after uboone colab meeting" |
| 2020-08-11 | `261f47b` "valid DL vtx" in `pid` | |
| 2020-08-17/18 | last training-repo commits; copied to `HaiwangYu/nue-cc-dnn-vtx2` (2020-08-18) | `8108f05a`, `b24eb601` |
| 2021-09-17 | README touch — repo dormant since | `c9716bb8` |
| **2026-04-05** | the same `.pth` lands in **our** `wire-cell-data` as `uboone/scn_vtx/…` | `979a156` "add files for uboone" |

`BNLIF/wire-cell-pydata` holds exactly two payloads — `scn_vtx/t48k-m16-l5-lr5d-res0.5-CP24.pth`
(28 759 205 B) and `scn_vtx/nuecc-sample.npz` (27 550 B, "simulation from `nue-6972-54-2707.root`").

**Byte-identity, proven.** The git blob SHA-1 of GitHub's copy and of ours are the same object:

```
$ gh api repos/BNLIF/wire-cell-pydata/git/trees/master?recursive=1 \
    --jq '.tree[]|select(.path|endswith(".pth"))|[.path,.size,.sha]|@tsv'
scn_vtx/t48k-m16-l5-lr5d-res0.5-CP24.pth  28759205  a18950a99d537f31ddbb5aad08c1f0b371bb9519
$ git hash-object wire-cell-data/uboone/scn_vtx/t48k-m16-l5-lr5d-res0.5-CP24.pth
a18950a99d537f31ddbb5aad08c1f0b371bb9519
$ md5sum wire-cell-data/uboone/scn_vtx/…CP24.pth prototype_base/input_data_files/scn_vtx/…CP24.pth
9cc1413e053c09534edc2d37cdfdc1d4  (both)
```

**So the net we run in SBND production today is bit-for-bit the file Haiwang published on
2020-08-06** — the prototype runs the same bytes, and it has never been retrained. Any
train/inference distribution mismatch that existed in August 2020 is still fully in force.

### 2.2 What the training sample was

From `train1.py` (defaults) and `.gitignore`:

* `--train-list list/nuecc-39k-train.csv`, `--val-list list/nuecc-21k-val.csv`;
  `hybrid_vtx.py`/`explore.py` also use `list/numucc-24k-val.csv`.
* `.gitignore` hides `list/*.csv`, `t100 t500 t16k t48k numu18k numucc*`, `checkpoints`,
  `work-model*` — i.e. **the file lists and the checkpoints were never committed**; the repo has
  only one branch (`master`) and no `list/` ever appeared in it.
* `util/loader.py` reads the sample straight out of WCP production ROOT files
  (`uproot.open(meta[0])`), with the truth vertex carried as columns 2/3/4 of each CSV row.
* A commented-out debug line pins the working area: `np.savez('/home/yuhw/wc/nue-cc/tmp.npz', …)`.

> **Answer to Q2 (dates).** Haiwang's training directory is `HaiwangYu/uboone-dl-vtx`, created
> **2020-06-10**, with all model development between then and **2020-08-18**; the production
> weights were frozen and published on **2020-08-06**. The training ROOT files therefore existed
> **before 2020-06-10** — they were *inputs*, not products, of that repo.

---

## 3. What the net was actually trained on — and why exclusion could matter at all

This is the part that makes the owner's hypothesis well posed rather than a category error.

**Training input** (`uboone-dl-vtx/util/loader.py`, `load()`), per event:

```python
tblob = root_file['T_rec_charge_blob']      # x, y, z, q  -> blob_coords, blob q
tvtx  = root_file['T_vtx']                  # x, y, z, type, flag_main -> vtx_coords
...
coords = np.concatenate((vtx_coords, blob_coords), axis=0)
ft     = np.stack((q, 0, 0, dist2prob(blob_coords, true_vertex, sigma)))
```

then `voxelize(..., resolution)` at 1.0 cm early, **0.5 cm** for the shipped model (`res0.5`).
`train1.py` sets **`nIn = 1`** — the network sees **charge only**; the `type`/`flag_main` columns
are carried but not fed. Loss is `nn.MSELoss` against a Gaussian-in-distance target.

**In the app that made Haiwang's files, `T_rec_charge_blob` is not imaging blobs.** In the production app
the tree (`pid/apps/wire-cell-prod-nue.cxx:3207`) is filled at `:3227` by

```cpp
neutrino_vec.at(i)->fill_skeleton_info(mother_cluster_id, point_tree, t_rec_deblob,
                                       dQdx_scale, dQdx_offset, /*flag_skip_vertex=*/true);
```

and `NeutrinoID::fill_skeleton_info` (`pid/src/NeutrinoID.cxx:2004`) writes
`vtx->get_fit_pt()` / `seg->get_point_vec()` positions with
`reco_dQ = dQ * dQdx_scale + dQdx_offset` (`dQdx_scale = 0.1`, `dQdx_offset = −1000`).
`T_rec_charge` is the sibling tree, filled by `fill_skeleton_info_magnify`
(`NeutrinoID.cxx:1852`) from the same quantities — `vtx->get_fit_pt()` / `seg->get_point_vec()`
with `dQ·0.1 − 1000` — differing only in that it loops per cluster, keeps vertices (with a
`map_vertex_in_shower` filter) and carries the residual-range branch. The training loader takes
the vertex-free variant and re-adds vertices from `T_vtx`.

**The tree name alone does not settle this — the *app* that wrote the file does.** `t_rec_deblob`
/ `T_rec_charge_blob` is also written by the *imaging* apps, where it genuinely is a blob charge
cloud (`x_save/charge_save`, no fitting involved) and would be exclusion-**in**sensitive. The
discriminator is that Haiwang's loader also needs `T_vtx` with `type` **and** `flag_main`, which is
a NeutrinoID product. Scanning every `T_rec_charge_blob` writer in the prototype:

```
$ for f in $(grep -rl 'T_rec_charge_blob' --include=*.cxx . | grep -v no_support); do
    printf "%-64s T_vtx=%s flag_main=%s fill_skeleton_info=%s\n" $f \
      $(grep -c '"T_vtx"' $f) $(grep -c flag_main $f) $(grep -c fill_skeleton_info $f); done

pid/apps/wire-cell-prod-nue.cxx                            T_vtx=1 flag_main=6 fill_skeleton_info=2
pid/apps/wire-cell-prod-nue-port.cxx                       T_vtx=1 flag_main=6 fill_skeleton_info=2
pid/apps/wire-cell-prod-nue-mt.cxx                         T_vtx=1 flag_main=6 fill_skeleton_info=4
pid/apps/wire-cell-prod-pi0.cxx                            T_vtx=1 flag_main=6 fill_skeleton_info=2
pid/apps/wire-cell-prod-nnbar.cxx                          T_vtx=1 flag_main=6 fill_skeleton_info=2
uboone_nusel_app/apps/wire-cell-imaging-lmem.cxx           T_vtx=0 flag_main=0 fill_skeleton_info=0
uboone_nusel_app/apps/wire-cell-imaging-lmem-celltree.cxx  T_vtx=0 flag_main=0 fill_skeleton_info=0
uboone_nusel_app/apps/…-celltree-porting.cxx               T_vtx=0 flag_main=0 fill_skeleton_info=0
uboone_nusel_app/apps/wire-cell-error-validation-celltree.cxx  T_vtx=0 flag_main=0 fill_skeleton_info=0
uboone_eval_app/apps/{wire-cell-truth-imaging,…-eval,…-eval-celltree}.cxx  T_vtx=0 flag_main=0 …=0
2dtoy/apps/{truthMC,toffset,milind,mooney,mooney-2plane}.cxx              T_vtx=0 flag_main=0 …=0
dune_app/apps/{dune_work_space,35ton-disambiguity}.cxx                    T_vtx=0 flag_main=0 …=0
```

**Every** app that can produce a file the loader can open is a `pid/apps/wire-cell-prod-*`
NeutrinoID app, and **every one of those** fills `T_rec_charge_blob` from `fill_skeleton_info`.
No app writes `T_vtx` *and* a genuinely blob-filled `T_rec_charge_blob`. (The imaging apps'
version of the tree is untouched since 2018-11-28 / 2022 in the superproject history.)

**Inference input** (`pid/src/NeutrinoID_DL.h:16-33`): vertices first
(`vtx->get_fit_pt()`, `vtx->get_dQ()*0.1 − 1000`), then segment interior points
(`pts.at(i)`, `dQ_vec.at(i)*0.1 − 1000`); `pyinf/SCN_Vertex.py` voxelizes at 0.5 cm with `nIn=1`.
The toolkit does the identical thing (`clus/src/NeutrinoVertexFinder.cxx:4851-4889`,
`dQdx_scale=0.1` / `dQdx_offset=-1000` in `TaggerCheckNeutrino.cxx:884-885`); pr/108 §1 already
records "DL input cloud … identical" between the two implementations.

> **Consequence.** Training and inference consume the **same object**: the *fitted trajectory*
> point cloud carrying the dQ/dx-fit charge. That cloud is a direct product of
> `do_multi_tracking(..., flag_exclusion)`. **The exclusion fit does shape the training
> distribution** — so "the net was trained on images without the exclusion fit" is a physically
> meaningful hypothesis, and worth the dating exercise below.

---

## 4. The dating constraint that (nearly) closes it

The loader needs **both** `T_rec_charge_blob` **and** `T_vtx` with `type` + `flag_main`, and §3
showed that only the `pid/apps/wire-cell-prod-*` NeutrinoID apps satisfy that pair — so the search
is correctly confined to the `pid` repo (the imaging apps' `T_rec_charge_blob` goes back to
2018-11-28 but those files have no `T_vtx`, hence cannot be the training input).
`git log -S 'T_rec_charge_blob' -- apps/` over the whole `pid` repo returns, oldest first:

```
b0d46f5 2020-03-17 catch up          <-- earliest appearance anywhere in pid/apps
f7bca39 2020-04-06 update output
50c06ac 2020-04-13 catch up
ca682ce 2021-11-02 …                 (later, unrelated apps)
```

The 2020-03-17 snapshot already has everything the loader needs:

```
$ git show b0d46f5:apps/wire-cell-prod-nue.cxx | grep -nE 'T_rec_charge_blob|"T_vtx"'
1328:  TTree *T_vtx = new TTree("T_vtx","T_vtx");
1477:  TTree *t_rec_deblob = new TTree("T_rec_charge_blob","T_rec_charge_blob");
# and T_vtx->Branch on x, y, z, type, flag_main, cluster_id, sub_cluster_ids  (lines 1341-1347)
```

And the last exclusion-free *release*, `v00_13_03` (pid `7921bfb9`, 2020-01-22), has **neither**
tree — `grep -c` returns 0 for `T_rec_charge_blob`, `T_vtx` and `flag_main` in its
`wire-cell-prod-nue.cxx`. (The app itself was only created 2020-02-13, `6ab2cdd`.)

Putting the two timelines side by side:

```
2020-01-22   v00_13_03 pid snapshot ......... no T_vtx, no T_rec_charge_blob, no exclusion
2020-02-13   wire-cell-prod-nue.cxx created . T_vtx appears
2020-03-17   b0d46f5 ...................... T_rec_charge_blob appears  <-- earliest possible sample
             |
             |  <-- 7 days: the ONLY window with the trees but no exclusion
             v
2020-03-24   c16dce5 + e659ad3 ............. exclusion born, 1 call site
2020-03-29   a41dccb ....................... 8 call sites
2020-05-11   d2e3587 ....................... 29 call sites (saturated)
2020-06-10   uboone-dl-vtx created ......... training starts (sample already on disk)
2020-06-17   v00_13_04 tagged .............. first production release with exclusion
2020-08-06   weights frozen and published
```

**Reading.** For the shipped net to have been trained on a *fully* exclusion-free cloud, the
39k + 21k (+ 24k numuCC) MicroBooNE sample would have to have been produced from a build inside
2020-03-17 … 2020-03-24 and then left unused for roughly three months while the production code
moved on by ~470 `pid` commits. Possible, but implausible. The natural reading is a sample
produced somewhere in **April–June 2020** from a then-current dev build (there is no tag between
`v00_13_03` and `v00_13_04`, so this was necessarily an untagged build), which means:

* produced **late March – early April 2020** ⇒ **partial** exclusion, 1–8 of 29 call sites;
* produced **May 2020 or later** ⇒ **full** exclusion, as in production today.

**Neither branch is excluded by the evidence at hand, but "no exclusion at all" effectively is.**

### 4.1 What this does *not* rule out

* A **partial-rollout** training distribution (late-Mar/early-Apr build). The first eight call
  sites are concentrated in `find_other_segments` and the early `find_proto_vertex` passes; a
  sample from that build would carry exclusion in some passes and not in the final ones. This is a
  real, unresolved possibility and would still be a train/infer mismatch today.
* A **prototype-vs-toolkit** mismatch — see §5. This one we have positive evidence for.

---

## 5. Two live explanations for "DL vertex is better with exclusion OFF", ranked

**(B) — better evidenced — the toolkit's exclusion fit is not WCP's exclusion fit.**
The net was trained on *WCP* clouds, whatever their exclusion state. pr/108 and pr/109 show the
toolkit's exclusion does not reproduce WCP's effect on SBND:

* pr/108 §5 — on SBND the toolkit's exclusion-ON trajectory carries **13 % less charge within 1 cm
  of the target vertex** than OFF; on uBooNE both implementations move that charge *up*
  (WCP +3…+15 % on 9/11 junctions; WCT +18…+33 % on two events, 0…−4 % on three).
  The **sign flips between detectors on the toolkit side only**.
* pr/109 — measured-vs-predicted 2-D charge near the vertex: uBooNE exclusion-ON is *better* in
  **both** implementations (median ΔU −0.024 WCT / −0.035 WCP), while **SBND exclusion-ON is worse
  on 6/6 events** (+0.025, U and W planes).
* pr/107 — the toolkit has an extra, prototype-absent third `form_map_graph(flag_exclusion)` pass
  before the dQ/dx fit that *deletes* the junction points exclusion stripped (443 + 101 points on
  nueCC48); `dqdx_fit_keep_all_points` removes that toolkit-only drop.
* pr/106 §9 — feeding the DL net an exclusion-free cloud recovers exactly the stripped junction
  cells and moves the vertex metric 35 → 41 (and the surgical knob `dl_vtx_cloud_no_exclusion`
  35 → 38, but nue-selected 35 → 32, so it was not flipped).

Under (B) the DL-off gain is a **toolkit-side artefact of a harsher exclusion**, and the fix is to
make the toolkit's exclusion functionally identical to WCP's — not to retrain.

**(A) — still live — a partial-rollout (or, at long odds, exclusion-free) training distribution.**
§4 shows this can only be *partial*, not absent. Under (A) the fix is to retrain on a current
production, and no amount of WCP/WCT parity work would close the gap.

The two are **not mutually exclusive** and they predict the same *accuracy* observation. They are
separated by the test in §7.2.

---

## 6. Corroborating but non-probative

`t48k-m16-l5-lr5d-res0.5-CP24.pth` is a **bare `OrderedDict` state_dict in the legacy (non-zip)
torch serialization** — `file` reports `data`, `unzip -l` reports "End-of-central-directory
signature not found", and the pickle header shows `protocol_version` / `sparseModel.*` keys with
no epoch, optimizer, loss or dataset metadata. PyTorch made zipfile serialization the default in
**1.6 (July 2020)**, so this is consistent with a save from a pre-1.6 torch — i.e. on or before
~2020-07-28, which matches §2.1.

**Caveat: this is not proof.** `torch.save(..., _use_new_zipfile_serialization=False)` produces the
same thing on any later version, and `pyinf/SCN_Vertex.py` even carries a shim for shape
differences between torch 1.0.0 and 1.3.1. Treat as consistency, not evidence. The checkpoint
carries **no** embedded training metadata, so it cannot date itself.

---

## 7. What would settle it

### 7.1 Ask Haiwang (cheapest, decisive)

The one artefact that would close §4 is `list/nuecc-39k-train.csv` — it is gitignored, so it exists
only on his disk. Concretely worth asking:

1. Which MicroBooNE production made the ROOT files under `/home/yuhw/wc/nue-cc/`
   (release tag, or the `/pnfs/uboone/...` path), and roughly when were they generated?
2. Were the `t48k` files the same ones used for `t16k` (2020-07-06), or regenerated in between?

A release tag ≥ `v00_13_04`, or a production date ≥ May 2020, retires explanation (A) outright.

### 7.2 The distribution-mismatch signature (measurable here, no new infrastructure)

Accuracy improving with exclusion OFF is consistent with **both** (A) and (B). A *training
distribution* mismatch has a distinct extra signature: the SCN's **peak score / confidence** should
be systematically higher on no-exclusion clouds — not merely correct more often. If OFF wins on
accuracy but the score distribution is unchanged, that points at (B), the geometry of the cloud,
rather than at (A).

The instrumentation already exists: `dl_vtx_harvest` (pr/79 §10) captures the exact live SCN input
cloud, and `PRVertexScoreboard` (`clus/inc/WireCellClus/PRVertexScoreboard.h`) records the
per-candidate DL scores; `dl_vtx_cloud_no_exclusion` (pr/106 §9, default OFF) supplies the
exclusion-free arm. Comparing the **peak-score distributions** of the two arms over nueCC48 is a
single A/B pair with no new code.

**Not run in this round** — flagged for the owner to direct.

---

## 8. Status

* **Doc only.** No C++, no jsonnet, no knob, no default changed. Nothing to gate; the byte-identical
  bar does not apply because no code path was touched.
* Every date, commit hash and count above is reproducible from §0. All commands are read-only;
  no repository was cloned, fetched or modified (GitHub was queried through `gh api`).
* Deliberately **not** claimed: which production release made Haiwang's training sample. The
  evidence brackets it (≥ 2020-03-17, ≤ 2020-06-10, no tag in that window) but does not pin it.
