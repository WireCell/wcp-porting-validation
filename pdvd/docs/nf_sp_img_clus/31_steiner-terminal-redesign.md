# Steiner terminal determination: what the algorithm does, where it is uBooNE-shaped, and a redesign

Continues [28_steiner-terminal-charge-pdvd-vs-sbnd.md](28_steiner-terminal-charge-pdvd-vs-sbnd.md)
(which decomposed the 500-vs-4000 e floor into a wire-crossing geometric
mismatch) and answers doc 26 §8 item 1, which sent 039349/14 to "a separate
Steiner-terminals campaign". Doc 26 §8 also ordered over-clustering first and
Steiner terminals after; the owner has opened this thread directly, which
supersedes that ordering. Doc 26's *other* case, 039349/53, was closed by
doc 27 as a stale-geometry face swap, not over-clustering.

**Scope.** Round 1 examined the current algorithm, designed an alternative and
reported a feasibility study, changing no code and no config. Round 2 ran the
per-phase census and added one **env-gated, log-only** terminal dump
(`WCT_STEINER_PHASE_DUMP`, C++ default OFF). Round 3 **fixed** the bug round 2
found, behind a second default-OFF knob (`wrapped_channel_charge`), and measured
that it works and does **not** cure the symptom. **Round 4 (§8) answers the
owner's four questions and, in doing so, root-causes the symptom** — a third site
of §5's `AnodePlane` rule, inside the Steiner stage's own retiler, which the
round-3 fix structurally could not reach. Round 4 changes no behaviour: one
doctest plus three fields on round 2's env-gated dump line.
**Round 5 (§9) fixes that site behind a second default-OFF knob
(`wrapped_channel_activity`) and closes the 108.5 cm gap to 1.8 cm.**
**Round 6 (§10) is the owner's hand scan and the three decisions it settled**:
both knobs go to PDVD production (the two downstream STM verdict changes are
adjudicated in the fix's favour), `ImproveCluster_2`'s terminal threshold is
synced to `CreateSteinerGraph`'s, and `traj_degenerate_wcpts_fallback` is
retired. Its one negative verdict — the terminals are now too dense — is
root-caused in §10.3 and becomes round 7.
Gates in §5.3, §8, §9.2 and §10.5; `./build/clus/wcdoctest-clus` **277/277**.

---

**TL;DR — SOLVED.** `IWirePlane::channels()` is a channel *list*; three separate
places in this codebase used it as a wire→channel *lookup table*, and each failed
silently on exactly the wrapped strips PDVD is full of. Fixing two of them
(rounds 3 and 5) takes the starved half of 039349/14 from **5 steiner points and
1 terminal** to **666 and 198**, and the largest steiner-free gap from
**108.5 cm to 1.8 cm**. Both fixes are needed and they repair different planes:
the retile fix (round 5) recovers U, round 3's sampler fix recovers V (§9.3).

Round 2 overturned round 1's conclusion; round 3 fixed a real bug and was
overturned in turn on the question that mattered; round 4 found why; round 5
fixed it. Current picture:

0. **The terminal starvation is root-caused, and it is a bug, not a criterion**
   (§8.2). The Steiner stage runs on a **retiled** cluster, and the retiler
   (`improvecluster_1.cxx:840`) indexes the plane's channel vector by *wire
   index*. On the 16 PDVD planes carrying wrapped continuations that vector is
   shifted — completely, for the 8 planes whose continuation band starts at wire
   index 0. Measured in the retiled cluster along the starved stretch: U ≠ 0 on
   **0.000** of points, V ≠ 0 on **0.004**, and **99.8 %** carry fewer than the
   two non-zero planes `calc_charge_wcp` requires, so their charge is 0 and they
   cannot be candidates **at any threshold**. Above the vertex, 21.7 % carry two
   — and Phase 1 returns 44 terminals there against 1 below. Predicted before
   measuring, including that W (never wrapped) must not move; it did not.
   **Fixed in round 5 (§9); the gap closes to 1.8 cm.**

1. **The terminal finder IS the defect, and the filters are innocent** (§4.1).
   Measured per phase inside `create_steiner_tree` on the real event: the
   cluster the Steiner stage actually runs on has **1239 points along the
   starved 111 cm, largest gap 2.4 cm** — dense, continuous coverage.
   `find_steiner_terminals` (Phase 1) returns **1** terminal there, against 44
   on the control half. Phase 2 then removes 16 of 406 cluster-wide and
   **Phase 3 removes nothing at all** (on every one of the 50 calls in the
   event). The terminals below V are never created; they are not filtered away.
   Round 1's §4 said the opposite — see §4.2 for why the inference failed.
2. **The V-plane zero-charge defect is root-caused and fixed** (§5.1-§5.3), and
   it was a real bug. `Gen::AnodePlane` builds `IWirePlane::channels()` by
   skipping every wire with `segment() > 0` (`AnodePlane.cxx:244-247`), so a
   plane does not list the channels of wrapped strips whose segment-0 half lives
   in the sibling face. `BlobSampler` looked that list up with
   `unordered_map::operator[]`, which **inserts 0 on a miss**, so those points
   silently took `channels[0]`'s activity — usually absent, leaving
   `charge_val` *and* `charge_unc` at exactly 0, which `calc_charge_wcp` reads
   as "no signal, don't hold it against the point". **1568 PDVD wires (11.3 %)
   and 6400 PDHD wires (28.8 %)**; SBND and uBooNE have no wrapped channels and
   are structurally immune. Fixed by resolving the charge by channel **ident**,
   as `add_ctpc` already does. Along the starved stretch, V-plane zeros go
   **98.6 % → 4.4 %**, and 44 genuinely dead channels stop masquerading as
   "no signal".
3. **The fix does not bring the terminals back** (§5.3). Same event, one knob
   apart: the largest steiner-free gap below V stays at **108.5 cm**, Phase 1
   still returns **1** terminal there, and the tree is unchanged. Round 2's
   recommendation — that fixing the charge might alone make the redesign
   unnecessary for this event — is **refuted by measurement**. Correcting the
   charge in fact made the candidacy gate *stricter* below V (58.3 % → 50.9 %),
   because `(q > cut) || (q == 0)` had been letting the zeroed plane pass for
   free.

(2) looked like the cause of (1) — it deletes an entire induction plane from
every point along precisely the stretch where (1) fails to make terminals, and
the criterion is charge-based. (3) is the test of that hypothesis, and it fails
it. **(0) is why**: (2) and (0) share a root cause but not an object. (2) is in
`BlobSampler`, and it corrupts the *persisted* point cloud that clustering, Q/L
and the taggers read. (0) is in the *retiler*, which throws that point cloud's
charges away and rebuilds an activity map from the ctpc clouds — so the round-3
knob is not merely ineffective on the Steiner stage, it is structurally unable to
reach it. Both are real; only (0) explains (1).

Everything reproduces on doc 27's fresh, fully self-consistent v7 arm, so none
of it is an artifact of the v6/v7 mixing doc 27 found.

**What this does and does not do to §6.** The redesign's case on *this event* is
suspended, not withdrawn: until (0) is fixed, 039349/14 cannot say whether the
criterion is wrong, only that it was starved. The case from doc 28's
**population** statistics and §7's aperture measurement is untouched — both were
measured on the input point cloud, which the retile defect never touches. §5.3's
mechanism argument also survives and is in fact strengthened: `(q > cut) ||
(q == 0)` cannot tell "no charge" from "no readout", and (0) is the third
distinct time that conflation has produced a silent wrong answer here. §7
measures §6's core idea — combine nearby wires over a fixed **physical** aperture instead
of reading one snapped wire — which removes essentially all of the
peak-vs-losing candidate asymmetry doc 28 identified (losing/peak pass-rate
ratio 0.77 → 0.92-1.00 on PDVD, 0.68 → 0.93-1.00 on SBND). The next round is
**fix (0), then re-measure the 108.5 cm gap, then reopen §6** — see §8.5 and §9.

---

## 0. Repro block

```bash
cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdvd

# Sections 4 and 5 -- stage attribution and the V-plane defect.
# d27fresh is doc 27's self-consistent v7 arm (imaging+clustering+PR on one
# geometry); d25r13fix is the older mixed arm, kept as a cross-check.
python3 docs/nf_sp_img_clus/scripts/steiner_terminal_attribution.py work/039349_14_d27fresh
python3 docs/nf_sp_img_clus/scripts/steiner_terminal_attribution.py work/039349_14_d25r13fix

# Sections 4.1 / 5.1 (round 2) -- the per-phase terminal census.
# The per-phase COUNTS are already TRACE lines in create_steiner_tree; they
# only need the level raised, no code change:
mkdir -p work/039349_14_d31phase
cp -n work/039349_14_d27fresh/pctree-evt19689.{tar.gz,tlas} work/039349_14_d31phase/
PDVD_KEEP_CFG=1 PDVD_PR_COMPILE_ONLY=1 ./run_pr_evt.sh -s d31phase -stm-fit 039349 14
cd work/039349_14_d31phase
wire-cell -l stderr -l "wct_pr_trace.log:trace" -L clus:trace -c .wct-pr_d31phase.json
# The per-phase POSITIONS need the env-gated dump (toolkit, default OFF):
WCT_STEINER_PHASE_DUMP=1 wire-cell -l stderr -l "wct_pr_dump.log:trace" \
    -L clus:trace -c .wct-pr_d31phase.json

# Section 5.1 -- the wrapped-strip root cause.  Needs the wires file's
# channel/segment map; note the schema references faces/planes/wires by ARRAY
# INDEX, not by `ident` (idents are local and repeat).
python3 docs/nf_sp_img_clus/scripts/steiner_wrapped_channel_census.py \
    work/039349_14_d27fresh/pctree-evt19689.tar.gz

# Sections 5.2 / 5.3 (round 3) -- the wrapped-segment fix.
# The orphan census needs only the wires files (no event, no run):
python3 docs/nf_sp_img_clus/scripts/steiner_orphan_channel_census.py

# The two end-to-end arms.  THREE non-obvious prerequisites, each of which
# fails loudly-but-late if you skip it:
#   * PDVD_LIGHT_SUFFIX=_keep -- the light for this event is under
#     work/039349_light19689_keep, not the bare name the runner defaults to;
#     without it Q/L is skipped, there is no t0, and x_t0cor is garbage.
#   * img-provenance.txt must be carried with the imaging tarballs, or doc 27's
#     provenance guard refuses the inputs (rc=2, and the wire-cell log is never
#     even created).
#   * LD_LIBRARY_PATH pins a snapshot of local/lib: this is a shared tree and a
#     peer's `wcbuild` mid-arm silently swaps the binary under you.
# The knob must be set at CLUSTERING time: flipping it in the PR job alone is
# byte-identical (verified).  See section 8.2's round-4 correction for WHY --
# the retile does re-sample, but its activity map is keyed from `channels`
# itself, so the knob has nothing new to find there.
mkdir -p /home/xqian/tmp/d31r3lib && cp -a ../../local/lib/. /home/xqian/tmp/d31r3lib/
for arm in off on; do
  W=work/039349_14_d31fix2$arm; mkdir -p $W
  ln -f work/039349_14_d27fresh/clusters-apa-anode*-ms-*.tar.gz $W/
  ln -f work/039349_14_d27fresh/img-provenance.txt $W/
  [ $arm = on ] && K="-S wrapped_channel_charge=true" || K=""
  env LD_LIBRARY_PATH=/home/xqian/tmp/d31r3lib PDVD_LIGHT_SUFFIX=_keep \
      PDVD_CLUS_TLA="$K" PDVD_KEEP_CFG=1 \
      ./run_clus_evt.sh -s d31fix2$arm -save-pctree -calib 39349 14
  PDVD_KEEP_CFG=1 PDVD_PR_COMPILE_ONLY=1 PDVD_PR_TLA="$K" \
      ./run_pr_evt.sh -s d31fix2$arm 39349 14
  (cd $W && env LD_LIBRARY_PATH=/home/xqian/tmp/d31r3lib WCT_STEINER_PHASE_DUMP=1 \
      wire-cell -l stderr -l "wct_pr_dump.log:trace" -L clus:trace \
      -c .wct-pr_d31fix2$arm.json)
done

# The three round-3 measurements.
python3 docs/nf_sp_img_clus/scripts/steiner_wrapped_channel_census.py \
    work/039349_14_d31fix2on/pctree-evt19689.tar.gz      # zeros gone, W unmoved
python3 docs/nf_sp_img_clus/scripts/steiner_terminal_attribution.py \
    work/039349_14_d31fix2on                             # candidacy, ctpc match
python3 docs/nf_sp_img_clus/scripts/steiner_phase_census.py \
    work/039349_14_d31fix2off/wct_pr_dump.log \
    work/039349_14_d31fix2on/wct_pr_dump.log             # the per-phase census

# Compiled-config gate (knob OFF must equal HEAD byte-for-byte).  Compile the
# HEAD copies of clus.jsonnet / pr.jsonnet under a scratch WIRECELL_PATH and the
# HEAD copy of wct-pr-perevt.jsonnet beside the live one, then `cmp`.

# ---- round 4 (section 9) --------------------------------------------------
# 9.2 premise: the wire<->channel invariant, per detector and per plane, read
# through the SAME accessor AnodePlane walks.  Prints the per-plane rows the
# section quotes (287 wires / 98 continuations / first_bad 0 or 189).
cd /nfs/data/1/xqian/toolkit-dev/toolkit
./build/clus/wcdoctest-clus -tc="pdvd doc31 round4*" -s

# 9.1 + 9.2 measurement: the RETILED cluster's own per-plane charges and Phase
# 3's distance operands.  PR-only is enough -- the retile lives in the PR stage.
cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdvd
W=work/039349_14_d31r4dump; mkdir -p $W
ln -f work/039349_14_d31fix2off/pctree-evt19689.{tar.gz,tlas} $W/
PDVD_KEEP_CFG=1 PDVD_PR_COMPILE_ONLY=1 ./run_pr_evt.sh -s d31r4dump 39349 14
(cd $W && env LD_LIBRARY_PATH=/home/xqian/tmp/d31r4lib GOGC=off \
    WCT_STEINER_PHASE_DUMP=1 wire-cell -l stderr -l "wct_pr_dump.log:trace" \
    -L clus:trace -c .wct-pr_d31r4dump.json)
python3 docs/nf_sp_img_clus/scripts/steiner_retile_charge_census.py \
    work/039349_14_d31r4dump/wct_pr_dump.log

# 9.3 (owner Q2): event 298595 = run 039252 evt 2, knob OFF vs ON.  Same three
# prerequisites as the round-3 arms above; the knob goes on the CLUSTERING job.
for arm in off on; do
  W=work/039252_2_d31r4$arm; mkdir -p $W
  ln -f work/039252_2_d27fresh/clusters-apa-anode*-ms-*.tar.gz $W/
  ln -f work/039252_2_d27fresh/img-provenance.txt $W/
  [ $arm = on ] && K="-S wrapped_channel_charge=true" || K=""
  env LD_LIBRARY_PATH=/home/xqian/tmp/d31r4lib PDVD_LIGHT_SUFFIX=_keep \
      PDVD_CLUS_TLA="$K" PDVD_KEEP_CFG=1 \
      ./run_clus_evt.sh -s d31r4$arm -save-pctree -calib 39252 2
  env LD_LIBRARY_PATH=/home/xqian/tmp/d31r4lib WCT_DQDX_DROP_DEBUG=1 \
      ./run_pr_evt.sh -s d31r4$arm -stm-fit 39252 2
  python3 docs/nf_sp_img_clus/scripts/steiner_wrapped_channel_census.py \
      $W/pctree-evt298595.tar.gz
  grep "WCT_DQDX_DROP_DEBUG segment gi=2 cluster=86" $W/wct_pr_039252_2.log | head -2
  grep -oE "cluster [0-9]+ . STM=1" $W/wct_pr_039252_2.log | sort -u
done
cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img
diff <(python3 abtest/hash_archive.py --members pdvd/work/039252_2_d31r4off/mabc-pr.zip) \
     <(python3 abtest/hash_archive.py --members pdvd/work/039252_2_d31r4on/mabc-pr.zip)
cd pdvd

# ---- round 5 (section 9) --------------------------------------------------
# The fix's premise, and the knob default:
cd /nfs/data/1/xqian/toolkit-dev/toolkit
./build/clus/wcdoctest-clus -tc="pdvd doc31 round5*" -s

# Compiled-config gate.  This round edits the SHARED cfg/pgrapher/common/
# clus.jsonnet, so SBND and uBooNE must be proven untouched, not assumed:
#   PDVD  HEAD vs knob-OFF : byte-identical, 278613 B; ON adds one key, one node
#   SBND  HEAD vs new      : byte-identical, 253993 B
#   uBooNE HEAD vs new     : byte-identical, 255717 B
# (compile the HEAD copies of pr.jsonnet + common/clus.jsonnet under a scratch
#  WIRECELL_PATH prefix, and the HEAD runner beside the live one, then cmp.)

# The three arms.  PR-only: the retile lives in the PR stage and rebuilds its
# activity from the ctpc clouds, which are correct independently of round 3's
# knob -- so this is a clean single-knob change from production.
cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdvd
mkdir -p /home/xqian/tmp/d31r5lib && cp -a ../../local/lib/. /home/xqian/tmp/d31r5lib/
for arm in off on both; do
  W=work/039349_14_d31r5$arm; mkdir -p $W
  ln -f work/039349_14_d31fix2off/pctree-evt19689.{tar.gz,tlas} $W/
  case $arm in
    off)  K="" ;;
    on)   K="-S retile_wrapped_channel_activity=true" ;;
    both) K="-S retile_wrapped_channel_activity=true -S wrapped_channel_charge=true" ;;
  esac
  PDVD_KEEP_CFG=1 PDVD_PR_COMPILE_ONLY=1 PDVD_PR_TLA="$K" \
      ./run_pr_evt.sh -s d31r5$arm 39349 14
  (cd $W && env LD_LIBRARY_PATH=/home/xqian/tmp/d31r5lib GOGC=off \
      WCT_STEINER_PHASE_DUMP=1 wire-cell -l stderr \
      -l "wct_pr_dump.log:trace" -L clus:trace -c .wct-pr_d31r5$arm.json)
  python3 docs/nf_sp_img_clus/scripts/steiner_retile_charge_census.py \
      $W/wct_pr_dump.log                      # per-plane charge, section 9.3
  ln -f work/039349_14_d31fix2off/pctree-evt19689.tar.gz $W/
  python3 docs/nf_sp_img_clus/scripts/steiner_terminal_attribution.py $W
                                              # steiner points / terminals / GAP
done
# OFF-path end-to-end gate: d31r5off's calib-pr must equal round 4's exactly.
cmp work/039349_14_d31r4dump/calib-pr-evt19689.json \
    work/039349_14_d31r5off/calib-pr-evt19689.json

# ---- round 6 (section 10) -------------------------------------------------
# The three Bee hand-scan sets (no new reconstruction -- built from the arms
# above).  Prints the per-event layer point counts; upload each zip and record
# the UUID.  Set A carries the two NEW layers built from the calib steiner
# section.
cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/pdvd
python3 docs/nf_sp_img_clus/scripts/build_bee_sets.py /home/xqian/tmp/beesets
for z in A_density_039349_14 B_cl47_039349_14 C_cl109_039252_2; do
  ./upload-to-bee.sh /home/xqian/tmp/beesets/$z.zip
done

# The two round-6 arms.  prod = the new PDVD defaults; sync = owner Q5, the
# retiler's terminal finder at the same 500 e CreateSteinerGraph runs.
mkdir -p /home/xqian/tmp/d31r6lib && cp -a ../../local/lib/. /home/xqian/tmp/d31r6lib/
for arm in prod sync; do
  W=work/039349_14_d31r6$arm; mkdir -p $W
  ln -f work/039349_14_d31fix2off/pctree-evt19689.{tar.gz,tlas} $W/
  [ $arm = sync ] && K="-S retile_steiner_terminal_charge=500" || K="-S retile_steiner_terminal_charge=null"
  PDVD_KEEP_CFG=1 PDVD_PR_COMPILE_ONLY=1 PDVD_PR_TLA="$K" \
      ./run_pr_evt.sh -s d31r6$arm 39349 14
  (cd $W && env LD_LIBRARY_PATH=/home/xqian/tmp/d31r6lib GOGC=off \
      WCT_STEINER_PHASE_DUMP=1 wire-cell -l stderr \
      -l "wct_pr_dump.log:trace" -L clus:trace -c .wct-pr_d31r6$arm.json)
done
# Section 10.3's per-blob floor and section 10.6's density rows, both arms.
python3 docs/nf_sp_img_clus/scripts/steiner_density_census.py \
    work/039349_14_d31r5off work/039349_14_d31r6prod work/039349_14_d31r6sync
# Section 10.5's end-to-end gate: the default flip must reproduce round 5 exactly.
cmp work/039349_14_d31r5both/calib-pr-evt19689.json \
    work/039349_14_d31r6prod/calib-pr-evt19689.json

# Section 10.5's config gates.  The load-bearing one compiles the NEW tree with
# the sync suppressed against the HEAD tree with both knobs passed as TLAs:
#   git archive HEAD cfg | tar -x -C $G/headcfg
#   git -C ../  show HEAD:pdvd/wct-pr-perevt.jsonnet > wct-pr-perevt-head.jsonnet
#   WIRECELL_PATH=$TOOLKIT/cfg:$WCDATA     wcsonnet ... -S retile_steiner_terminal_charge=null wct-pr-perevt.jsonnet
#   WIRECELL_PATH=$G/headcfg/cfg:$WCDATA   wcsonnet ... -S wrapped_channel_charge=true -S retile_wrapped_channel_activity=true wct-pr-perevt-head.jsonnet
#   cmp                              # byte-identical, 279974 B
# SBND 253993 B and uBooNE 255717 B are gated the same way (shared common/clus.jsonnet).

# Section 7 -- aperture feasibility, PDVD and the SBND control.
python3 docs/nf_sp_img_clus/scripts/steiner_aperture_feasibility.py \
    work/039252_2_stm1/pctree-evt298595.tar.gz 500
python3 docs/nf_sp_img_clus/scripts/steiner_aperture_feasibility.py \
    /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin/work-dbg25a-d97off/ql_evt16/pctree-evt16.tar.gz 4000
```

The Steiner coverage and gap metric of §4 are section 6 of that same script.
They come from the calib dump's `steiner` section, which carries per-cluster
`x/y/z` **and** `flag_terminal`. The cluster is resolved by **ownership of the
control region**, never by a hardcoded id — re-clustering renumbers it (36 on
`d25r13fix`, 34 on `d27fresh`), and a hardcoded id would silently select nothing.

Four traps that each produced a wrong number first; all four are commented in
`steiner_terminal_attribution.py` and enforced by assertions where possible:

- Compare against PR-log / calib coordinates using **`x_t0cor`**, not `x`. With
  `x` the nearest sampled point to V is 16 cm away and every selection is empty.
  `x_t0cor` is meaningless (order 1e9) for points whose cluster has no t0, so
  never min/max it globally.
- The flat `live/.../namedpcs/3d` cloud is a **blob-ordered concatenation**:
  `np.repeat(arange(n_blobs), scalar/npoints)` gives exact blob attribution.
  `sum(npoints) == n_points` is the assertion that this still holds.
- The Bee `clustering-global` layer has the same point *count* as the 3-D cloud
  but **not the same order** (max |y_bee·10 − y_pc| = 5266 mm). Select
  geometrically in Bee's own (y, z) cm coordinates; never index one with the
  other's mask.
- Identify the `ctpc_a<A>f<F>p<P>` dataset by **charge-matching** on points with
  known nonzero charge, not by decoding `wpid` by eye. The by-eye guess gave
  `a2f1` and 0/140 agreement; the truth is `a4f0` at 194/194.
  `slice_index = (t/500)` floored to a multiple of 4. (The two agree once
  decoded properly: `wpid = layer | face<<3 | apa<<4`,
  `iface/src/WirePlaneId.cxx:5-7,35-36`, so 71 → apa 4, face 0.)

---

## 1. What the algorithm does today

`Steiner::Grapher::create_steiner_tree` (`clus/src/SteinerGrapher.cxx:22-143`),
called from `CreateSteinerGraph.cxx:283` as

```cpp
sg.create_steiner_tree(src, path_point_indices, "ctpc_ref_pid", "steiner_graph", false, "steiner_pc");
```

so `disable_dead_mix_cell = false` — which selects the second branch of
`calc_charge_wcp` throughout. Five phases:

| phase | what | source |
|---|---|---|
| 1 | `find_steiner_terminals` — per-blob charge-peak finding | `:38`, `:599-622` |
| 2 | `filter_by_reference_cluster` — drop terminals not contained in the reference cluster's wire ranges | `:50-59`, `:149-206` |
| 3 | `filter_by_path_constraints` — drop terminals far from the supplied path | `:65-70`, `:230-319` |
| 4 | extreme points inserted **unconditionally**, after both filters | `:75-79` |
| 5 | `create_enhanced_steiner_graph` — PAAL Voronoi tree over the survivors | `:111-113` |

### 1.1 Terminals are found one blob at a time

```cpp
for (const auto& [blob_node_idx, point_indices] : cell_points_map) {
    auto blob_peaks = find_peak_point_indices(point_indices, graph_name, disable_dead_mix_cell);
    steiner_terminals.insert(blob_peaks.begin(), blob_peaks.end());
}
```
(`SteinerGrapher.cxx:611-616`; prototype `PR3DCluster_steiner.h:719-726`.)

Each blob is an independent peak-finding universe, so **every blob holding at
least one candidate contributes at least one terminal**. That property is what
makes §4's measurement possible.

### 1.2 The charge quantity

`Cluster::calc_charge_wcp` (`Facade_Cluster.cxx:1031-1112`; prototype
`PR3DCluster_steiner.h:955-1025`) reads, per plane, the charge of the **single
wire nearest the point**, stamped onto the point at sampling time by
`BlobSampler` (`BlobSampler.cxx:315-370` → `ucharge_val`/`vcharge_val`/
`wcharge_val`). With `disable_dead_mix_cell = false`:

```
flag_p  = (charge_p > cut) || (charge_p == 0)      // zero passes: no signal is not held against you
charge  = sqrt( sum(charge_p^2 over planes with charge_p != 0) / n_nonzero ),  0 unless n_nonzero > 1
quality = flag_u && flag_v && flag_w
```

### 1.3 Candidacy and local-maximum suppression

```cpp
const double charge_threshold = m_config.terminal_charge_threshold;   // 4000 default
auto [charge_quality, charge] = m_cluster.calc_charge_wcp(point_idx, charge_threshold, disable_dead_mix_cell);
if (charge > charge_threshold && charge_quality) candidates_set.insert({charge, point_idx});
```
(`SteinerGrapher.cxx:432-442`.) Note the same cut is applied **twice** — once
per plane inside `calc_charge_wcp`, once to the plane-RMS.

Candidates are then walked in descending charge order and suppressed against
their **1-hop** graph neighbours (`nlevel = 1`, `SteinerGrapher.h:190,193`), and
directly edge-connected surviving peaks are merged by connected components,
keeping the peak nearest each component's centroid (`:459-593`).

One subtlety worth recording: `map_index_charge` is built from **one blob's**
points while the BFS runs over the **whole cluster graph**, so cross-blob
neighbours are silently skipped by the two
`map_index_charge.find(...) == end() continue;` guards (`:496`, `:513`).
"Locally highest" therefore means *highest among graph-adjacent points of the
same blob*. Ties never veto (only strict `<` clears `flag_insert`). Both match
the prototype (`PR3DCluster_steiner.h:820`, `:834`).

---

## 2. Where the algorithm is uBooNE-shaped

The philosophy — *the backbone is where the charge is locally highest* — is
sound and detector-independent. Four implementation choices are not.

| # | choice | why it is uBooNE-shaped | PDVD number |
|---|---|---|---|
| 1 | **absolute floor of 4000 e**, per plane *and* on the RMS | an electron count tuned to 3 mm pitch; charge per wire scales with pitch, and the absolute scale also depends on a charge calibration PDVD does not yet have (doc 28 §4.2) | PDVD runs 500 e; W-plane per-point median ~1400 e (doc 25 §13.4 item 8) |
| 2 | **AND over exactly three planes** | with uBooNE's symmetric ±60°/0° *and equal* 3 mm pitch, at most one plane is unlucky for any track direction; PDVD keeps the angles but not the pitches | U/V 7.65 mm, W 5.10 mm — 2.55× and 1.70× coarser |
| 3 | **"local" = 1 graph hop** | a hop's physical size depends on point density, which depends on the crossing ambiguity | PDVD emits a median of 4 points per (U,V) crossing vs SBND's 1, so a hop is *shorter* on PDVD exactly where the sampling is worst |
| 4 | **single snapped wire per plane** | one wire is a fair estimate of the local charge only when the pitch is fine compared with the deposit's transverse spread | transverse diffusion is 2.01 mm = **0.39 of the W pitch**, 0.26 of U/V (doc 25 §8): the charge lands on one or two wires and the observable is adjacent-wire *sharing* |

Choices 2 and 4 are the ones the owner's brief targets, and the toolkit itself
already documents the geometric assumption behind them. `BlobSampler.cxx:817-824`,
on the mid-plane pitch correction used to place every sampled point:

> This gives a relative pitch distance measured in the "mid" view that is half
> the distance between crossing point of the 0-rays and the 1-rays in the other
> two views. In general, this is **NOT** the same as the magnitude of "adjust" /
> "ac" vector above as that diagonal of the min/max parallelogram is not
> necessarily parallel to the pitch direction in the third, "mid" view. The two
> directions are **accidentally coincident for symmetric wire patterns like in
> MicroBooNE**.

That is an in-repo, source-level statement that the geometry the sampler assumes
is exact for MicroBooNE and approximate elsewhere — stronger evidence for the
owner's hypothesis than any measurement added here.

### 2.1 `ChargeStepped` is not already this

The tree contains an unused second sampler, `ChargeStepped`
(`BlobSampler.cxx:914-1272`), a port of WCP's `calc_sampling_points()`. Neither
detector uses it: `protodunevd/clus.jsonnet:197` and `sbnd/clus.jsonnet:206`
both select `stepped`. It *does* filter candidate wires by charge before forming
a crossing (thresholds 4000/4000/4000, `:938-940`), which would change the
ambiguity population — worth a config-only experiment on its own. But it is
**not** the "combine nearby wires" idea: at `BlobSampler.cxx:1214` it still
resolves the third plane to one wire,

```cpp
coordinate_t cother{smid.layer, static_cast<int>(std::round(pitch_relative))};
```

and reads that single wire's charge. So the estimator this doc proposes to
change is unchanged in `ChargeStepped`.

---

## 3. What has already been excluded

- **The two known port divergences are already on in PDVD.** `steiner_terminal_wire_tol=1`
  (the prototype's ±1 wire of slack in the terminal filter, doc pr/29 D1) and
  `steiner_terminal_adjacent_slice=true` (D12's dead t±1 branch) are set in
  `wcp-porting-img/pdvd/wct-pr-perevt.jsonnet:380-381`, committed `a61fa097`
  on 2026-09-02 15:56 — before every arm used here.
  pr/29 measured that with both off the toolkit discards 47.7 % of all Steiner
  terminals; PDVD is already paying the reduced version, and §4's drop is
  on top of that.

  > **Round-4 correction.** This paragraph used to end "SBND still runs both
  > **off**", and §10 repeated it as an open owner call. **Both were wrong when
  > written.** SBND has set `steiner_terminal_wire_tol = 1` and
  > `steiner_terminal_adjacent_slice = true` since `6ea51a3b` (2026-08-04,
  > *"…SBND ON (doc pr/29 D1/D12, SBND evt 388)"*),
  > `cfg/pgrapher/experiment/sbnd/wct-pr-perevt.jsonnet:381-382`. The detector
  > actually running the pre-prototype behaviour is **uBooNE**, whose config
  > names neither key (`qlport/uboone-mabc.jsonnet:1263` takes every signature
  > default, so both are suppressed and the C++ defaults 0 / false govern). See
  > §8.4 for the full per-detector table.
- **Stale geometry is not the explanation.** Doc 27 found that arms built from a
  v6 point tree and run against v7 anodes put clusters on anodes 2/3/6/7 one
  face height off in y. Cluster 36/34 here is on **anode 4, face 0**, which doc 27
  lists as unaffected — and every number in §4 and §5 was re-measured on doc 27's
  fresh self-consistent `d27fresh` arm and is unchanged.
- **It is not a cluster-boundary artifact.** The charge below V belongs to the
  same cluster: 735 of 795 Bee points (92 %) carry the track's own
  `real_cluster_id`, and 100 % do on the control half.

---

## 4. The stage attribution (round 1 measurement, round 1 conclusion WITHDRAWN)

Cluster 34 (`d27fresh`; 36 on `d25r13fix`) is one straight cosmic. Doc 26 §7.5
established that its Steiner cloud stops at the vertex V (x 273, z 87): coverage
on the half above, none along the 111 cm from V to A. Both halves of the *same
track* in the *same cluster* therefore form a controlled comparison.

Region selection is geometric: points within 3 cm of the V→A line (below) or the
V→U line (above), endpoints excluded.

| | **below V** (starved) | **above V** (control) |
|---|---|---|
| Bee points, and share owned by this cluster | 795, **92 %** | 180, **100 %** |
| sampled 3-D points | **721** | 195 |
| pass `calc_charge_wcp` + 500 e floor | **420 (58.3 %)** | 109 (55.9 %) |
| distinct blobs holding ≥1 candidate | **263** | 72 |
| ⇒ terminals Phase 1 must emit (lower bound) | **≥263** | ≥72 |
| terminals actually in the tree | **1** | 40 |
| steiner points actually in the tree | **5** | 258 |
| **survival, Phase 1 → tree** | **0.4 %** | **56 %** |
| largest steiner-free gap along the line | **108.5 cm** | 65.6 cm |

(`d25r13fix`, the older mixed arm, gives the same picture: 738 points, 419
candidates in 259 blobs, 1 terminal, 111.1 cm gap.)

The starved half has **3.7× more** candidate-bearing blobs than the half that
works, and ends with 1 terminal. Round 1 concluded from this that the filters
between Phase 1 and the tree were eating them. **That conclusion was wrong**;
§4.1 measured it directly and §4.2 says where the inference failed.

### 4.1 Round 2: the per-phase census, which names Phase 1

The counts were already in the code as `SPDLOG_LOGGER_TRACE` lines
(`SteinerGrapher.cxx:39,55,68,78`); they only needed the level raised. Positions
needed one env-gated, log-only dump (`WCT_STEINER_PHASE_DUMP`, default OFF).
Over all 50 `create_steiner_tree` calls in the event:

- **Phase 3 (`filter_by_path_constraints`) removed nothing at all** — not one
  terminal, in any call.
- **Phase 2 (`filter_by_reference_cluster`) removed 797 of 12131** cluster-wide,
  ≈ 6.6 %, with `wire_tol=1 adjacent_slice=true` confirmed in the log line.

For the call that builds our track's tree (identified by its output: 1726
steiner points and 394 terminals, matching the calib dump exactly):

| phase | terminals | below V | above V |
|---|---|---|---|
| cluster's own points (`P0`) | **5705** | **1275**, largest gap **0.7 cm** | 694 |
| P1 `find_steiner_terminals` | 406 | **1** | 45 |
| P2 `filter_by_reference_cluster` | 390 | **1** | 40 |
| P3 `filter_by_path_constraints` | 390 | **1** | 40 |

The cluster the Steiner stage actually runs on **covers the starved 111 cm
densely and continuously — 1275 points, largest gap 0.7 cm.** Phase 1 returns
**one** terminal from it. The filters then take 1 → 1.

**So the terminal finder is the defect, exactly where the owner's brief aimed,
and the redesign of §6 is aimed at the right stage after all.**

> **Round-4 refinement.** The *stage* is right; the *diagnosis* was not. §8.2
> measures the retiled cluster's per-plane charges — the quantity Phase 1's gate
> actually reads, which no earlier round could see because the retiled point
> cloud is never persisted — and finds that below V **99.8 % of its points hold
> fewer than the two non-zero planes `calc_charge_wcp` requires**. Phase 1 is not
> rejecting these points on a badly-shaped criterion; it is being handed charge 0
> for them, by a channel-indexing bug in the retiler itself. Read the "dense,
> continuous coverage" row above with that: the *points* are there, their
> *charges* are not.

### 4.2 Why round 1's inference failed — worth recording

Round 1 argued: 263 blobs below V hold a candidate, terminal finding is
per blob, therefore Phase 1 must emit ≥263 terminals. Each step is true of the
**input** point cloud. But `CreateSteinerGraph` **retiles** the cluster before
building the tree, and the retiled cluster is a different object: 5705 points
where the input cluster has 2348. The blobs Phase 1 iterates are not the blobs
that were counted. A lower bound derived on one point cloud was applied to
another.

The general lesson, and it is the same one doc 30 taught: an inference chain
over quantities that were never measured *at the site* is a hypothesis, not a
result. The 20-minute census that would have tested it was available from the
start — the TRACE lines already existed.

---

## 5. The V-plane defect: charge present in the 2-D map, zero on the point

Along the same stretch, cross-referencing each sampled point's stored per-plane
charge against the 2-D `ctpc_a4f0p*` (wire, slice) charge map from the same dump:

| plane | below V (n=721) | above V (n=195) |
|---|---|---|
| stored charge nonzero | **1.4 %** | 99.5 % |
| stored value == map value at the point's own cell | **0 / 721** | 194 / 195 |
| **stored 0 while the map holds charge there** | **677 (median 4979 e)** | 1 |
| `charge_unc == 0` (not the 1e10 dead sentinel) | **98.6 %** | 0.5 % |

U (709/721 exact) and W (709/721) match the map, so the point's (apa, face) and
wire indices are right — this is specific to V, and specific to this stretch.
The uncertainty is 0 rather than the dead sentinel, so `calc_charge_wcp` treats
V as "no signal, don't hold it against the point" (§1.2) rather than as dead:
the point keeps a healthy U/W RMS and still passes, which is why §4's candidate
count is high despite a whole plane being blank. The defect is silent by
construction.

### 5.1 Round 2: root-caused — the second segment of every wrapped strip

The discriminating test round 1 deferred, run:

| region | n | V wire is **segment 1** | `charge_val == 0` | agreement |
|---|---|---|---|---|
| below V (starved) | 721 | 711 (98.6 %) | 711 (98.6 %) | **1.000** |
| above V (control) | 195 | 1 (0.5 %) | 1 (0.5 %) | **1.000** |

**`charge_val == 0` ⟺ the point's induction wire is the second segment of a
wrapped strip.** Perfect separation, both directions, on 916 points — stored-zero
points are 100 % segment-1, stored-nonzero are 0 % segment-1.

Event-wide it is deterministic per (apa, face, plane), which is what makes it a
bookkeeping bug rather than a charge effect:

| plane | segment | n points | `P(charge==0)` |
|---|---|---|---|
| U | 0 | 45825 | 0.088 |
| U | **1** | 4362 | **0.563** |
| V | 0 | 44741 | 0.084 |
| V | **1** | 5446 | **0.582** |
| W | 0 (never wrapped) | 50187 | 0.041 |

and split by group, `P(charge==0 | segment==1)` is **1.000** for a4f0 V, a5f0 U,
a6f1 V, a5f0 V, a7f1 V (and 0.980 for a4f0 U). Five of twelve groups are exactly
deterministic. A charge fluctuation does not do that.

> **Round-3 correction.** An earlier version of this paragraph continued
> "…but ~0 for anodes 0-3, i.e. 1.000 on anodes 4-7 (top drift) and ~0 on
> anodes 0-3", and read that split as a property of the defect. **Withdrawn.**
> §5.2's geometry census shows there is no such asymmetry to explain: *every*
> anode 0-7 carries exactly 98 orphan U and 98 orphan V wires, in one of its two
> faces. The anodes 0-3 rows come from an offline segment label that this doc
> cannot vouch for — the label is read by indexing the wires file's raw
> `planes[].wires[]` array with the point's persisted `wire_index`, and the
> toolkit's `iplane->wires()` is that array after `WireSchema::load`'s
> correction pass and `AnodePlane.cxx:257`'s re-sort. If those orders differ for
> the bottom-drift anodes, the labels are wrong there and the rows mean nothing.
> **Not established either way, and not needed:** the rows above are kept only
> for anodes 4-7, where the mapping was verified independently (median distance
> from each point to the wire its `wire_index` names: **0.00 mm** as-read,
> 1446 mm under a swapped face — an exact match). Our track is on anode 4, in
> that verified set. Anodes 0-3 gave no clean match under any of the 16
> (apa, face) geometry groups; that is unexplained and is *not* claimed here as
> a defect.

**The site** is `BlobSampler.cxx:343-357`:

```cpp
IWire::pointer iwire = iwires[wire_index[ipt]];
channel_ident[ipt]   = iwire->channel();
channel_attach[ipt]  = p_chi2i[channel_ident[ipt]];   // operator[] on a miss -> 0
auto ich             = channels[channel_attach[ipt]];
auto ait             = activity.find(ich);
if (ait != activity.end()) { charge_val[ipt] = ...; charge_unc[ipt] = ...; }
```

`p_chi2i` is an `unordered_map<int,int>` (`BlobSampler.cxx:231`) built from
**this plane's** channel list. A wrapped channel is reachable from the face
holding one segment and, in the failing groups, not listed under the face
holding the other. `operator[]` on a missing key **inserts 0 and returns 0**, so
the lookup silently resolves to `channels[0]` — the plane's first channel, whose
activity is normally absent — and both `charge_val` and `charge_unc` stay
exactly 0. That is why it is invisible: `calc_charge_wcp` (§1.2) reads a zero
plane as *"no signal, don't hold it against the point"*, the point keeps a
healthy two-plane U/W RMS, and nothing warns.

Round 2 left one question open — whether the ident is *genuinely* absent from
`iplane->channels()`, or present with the failure elsewhere. **§5.2 answers it
from the source: genuinely absent, by construction.**

**Scale, and why it matters for §4.1.** Roughly 11 % of all sampled points in
this event (≈ 5600 of 50187) lose an entire induction plane silently. Along the
starved stretch it is 98.6 % of points. Phase 1's criterion is charge-based and
ANDs three planes, so it is being fed a systematically mutilated input over
exactly the region where it fails to make terminals. That is now the leading
candidate cause for §4.1 — **not yet proven**, because the retiled cluster's
per-point charges are not dumped, and the input-cloud candidacy rate stayed high
(58 %) with V zeroed. Establishing it is round 3's first task.

A converse class also exists, at the few-percent level: points whose stored
charge is nonzero but which do not match the map at their own cell (PDVD U,
10296 of 137264 nonzero points event-wide; SBND 0 of 23182). The expected
explanation — not verified here — is that `PointTreeBuilding::add_ctpc` drops
rows whose uncertainty exceeds the dead threshold (`PointTreeBuilding.cxx:296-299`)
while `BlobSampler` stores the value regardless, making the map a subset of what
the sampler saw. Whatever it is, it is the opposite direction from the V finding
and does not explain it. The apparent PDVD/SBND contrast here is **not**
decomposed and should not be read as a result: the event-wide counts were never
split into "absent from the map" versus "present but different", and SBND is a
single-anode dump where the (apa, face) attribution is trivial, so the two sides
are not measured the same way.

### 5.2 Round 3: the mechanism, read off the source

`Gen::AnodePlane::configure` builds each plane's channel vector by walking that
plane's wires and **skipping every wrapped continuation**
(`gen/src/AnodePlane.cxx:244-247`):

```cpp
IChannel::vector plane_channels;
for (auto w : wires) {
    if (w->segment() > 0) {          // <-- here
        continue;
    }
    const int chanid = w->channel();
    SimpleChannel* sch = chwcollector(chanid);
    sch->set_index(plane_channels.size());
    plane_channels.push_back(IChannel::pointer(sch));
}
```

That is a correct channel *list* — each channel appears once per anode, attached
to the plane holding its segment-0 wire, which is what `set_index` numbers. It
is **not** a wire→channel lookup table. `BlobSampler` used it as one.

So the defect fires on exactly the wires that are **orphans in their own plane**:
`segment > 0` **and** no segment-0 wire of the *same* plane carries the channel.
A strip that wraps back inside one plane is not an orphan — its channel is listed
via the segment-0 half and the lookup succeeds. Counting orphans straight from
each detector's production wires file
(`scripts/steiner_orphan_channel_census.py`, replicated as a doctest):

| detector | wires file | wires | wrapped chans | **orphan wires** | planes hit |
|---|---|---|---|---|---|
| **PDVD** | `protodunevd-wires-larsoft-v7-uvwfit` | 13856 | 1568 | **1568 (11.3 %)** | 16 of 48 |
| **PDHD** | `protodunehd-wires-larsoft-v1` | 22208 | 6400 | **6400 (28.8 %)** | 16 of 24 |
| SBND | `sbnd-wires-geometry-v0206` | 11276 | 0 | **0** | 0 of 6 |
| uBooNE | `microboone-celltree-wires-v2.1` | 8256 | 0 | **0** | 0 of 3 |

Three things follow.

1. **It is not a PDVD-only bug.** PDHD is affected 2.5× harder. PDHD's config
   never mentions this knob, so its C++ default is the whole of its protection
   (cf. `feedback_cpp_default_governs_only_silent_configs`). A PDHD flip is a
   separate owner call with its own gate, not a side effect of this one.
2. **SBND and uBooNE are structurally immune** — no multi-segment channel
   anywhere, so the lookup never misses, the fallback map is never built, and no
   setting of the knob can move them. That is the negative control, and it is
   what makes the byte-identity argument for those two detectors airtight rather
   than merely gate-tested.
3. **PDVD's split is uniform, not per-drift-side:** every anode 0-7 contributes
   98 orphan U + 98 orphan V wires in one of its faces (16 × 98 = 1568). This is
   what overturns §5.1's per-anode reading.

The failure is also worse than "charge becomes 0". `channels[0]` is a real
channel; when it happens to be live in that slice the point silently inherits
**another channel's charge**. On this event it was absent every time, so the
symptom was uniformly zero — but the mechanism admits a wrong non-zero value,
which no downstream sentinel could catch.

### 5.3 Round 3: the fix, and what it did *not* do

**The fix.** `BlobSampler` now resolves the charge by channel **ident** when —
and only when — the plane's index map misses:

```cpp
auto chit = p_chi2i.find(channel_ident[ipt]);
if (chit != p_chi2i.end()) { /* unchanged: index into channels[] */ }
else if (cc.wrapped_channel_charge) {
    channel_attach[ipt] = -1;                       // no index into THIS plane
    const auto& abi = activity_by_ident(islice);    // built lazily, cached per slice
    auto ait = abi.find(channel_ident[ipt]);
    if (ait != abi.end()) { charge_val = ...; charge_unc = ...; }
}
else { /* legacy reproduced exactly, operator[] insertion included */ }
```

This is not a new idiom. `ISlice::map_t` is *already* hashed and compared by
ident (`ISlice.h:36-48`), so an ident is a complete key, and
`PointTreeBuilding::add_ctpc` walks the same activity map **forward**
(`for (const auto& [ichan, charge] : activity)` → `ichan->wires()`,
`PointTreeBuilding.cxx:294-306`) and therefore never needs the reverse map at
all. That is precisely why the `ctpc_*` 2-D maps always carried the charge these
points were missing — §5's original observation, now explained.

Cost is zero where it cannot help: the ident map is built only after a miss, and
a detector with no wrapped channels never misses.

**Knob:** `wrapped_channel_charge`, `BlobSampler::CommonConfig`, **default
false**, threaded to PDVD through `clus.jsonnet`'s `bs_live_face` with the
key-suppression idiom. Applied to the **live** sampler (and hence the retile
sampler, `bs_rt_face = bs_live_face`) only; the dead-blob sampler is left alone
this round, deliberately — its charge feeds different logic and would need its
own gate.

**Gates.**

| gate | result |
|---|---|
| `./build/clus/wcdoctest-clus` | **273/273** (3 new cases; 270 before) |
| compiled PR config, knob OFF vs `HEAD` | **byte-identical**, 268489 B, md5 `e9b620c35d102d71ada99c5ef464ede3` |
| compiled PR config, knob ON | differs **only** by the key, on exactly 16 nodes = 8 anodes × 2 faces |
| PDVD 039349/14 end-to-end, knob OFF | `calib-pr` 7510870 B vs d27fresh's 7510869 — the one known timer byte |
| freshness | source 13:37:08 → `local/lib` 13:37:34 → test binary 13:38:36 |

The doctest pins the **default**, which is what protects PDHD; PDHD itself was
reasoned about from its wires file, **not** gated on an event.

**Verification that the fix works** — same event, same imaging inputs, one knob
apart (`work/039349_14_d31fix2{off,on}`):

| measurement, V plane along the starved stretch | knob OFF | knob ON |
|---|---|---|
| `charge_val == 0` | **711/721 (98.6 %)** | **32/721 (4.4 %)** |
| matches the `ctpc_a4f0pV` map exactly at its own cell | 0/721 | **677/721** |
| "stored 0 but the map has charge" | 677 (median 4979 e) | **0** |
| `charge_unc == 0` (the silent state) | 0.986 | **0.000** |
| `charge_unc > 1e10` (the honest dead sentinel) | 0.000 | **0.044** |

Event-wide `P(charge==0 | segment==1)`: U **0.563 → 0.052**, V **0.582 →
0.016**. **W is unchanged at 0.041 in both arms** — W is never wrapped, so it
must not move, and it does not. That is the internal negative control.

The `unc > 1e10` row is worth its own sentence: 44 of those points are on
genuinely **dead** channels, and the bug had been reporting them as
`val = unc = 0`, i.e. *"no signal"* rather than *"no readout"*. The fix restores
a sentinel, not just a number.

**And now the negative result, which is the point of this section.**

| | knob OFF | knob ON |
|---|---|---|
| retiled cluster (P0) | 5705 pts, 1239 below V | 5723 pts, 1239 below V |
| P1 `find_steiner_terminals` | 406 total, **1 below V**, 44 above | 407 total, **1 below V**, 44 above |
| P2 / P3 | 390, **1 below V** | 390, **1 below V** |
| steiner points reaching the tree, below V | 5 | 5 |
| terminals below V | 1 | 1 |
| **largest steiner-free gap below V** | **108.5 cm** | **108.5 cm** |
| candidacy below V (`calc_charge_wcp` + 500 e) | 420/721 = **58.3 %** | 367/721 = **50.9 %** |
| candidacy above V (control) | 109/195 = 55.9 % | 109/195 = **55.9 %** |

**Fixing the charge did not bring the terminals back.** The gap is identical to
the digit. Round 2's recommendation — "the honest possibility is that this alone
restores the terminals and no redesign is needed for *this* event" — is
**refuted by measurement.**

Two things make this a result rather than a null:

- **Phase 1 demonstrably saw the corrected charges.** Event-wide, terminals
  entering phase 2 go 12131 → **12127**. A small number, but a non-zero one: the
  stage's own output moved, so the correction reached it. (The earlier check
  that flipping the knob in the PR job *alone* left the calib dump
  byte-identical stands, and the fix does have to be applied at **clustering**
  time, where `pctree-evt*.tar.gz` is written — but the reason given here
  originally, "the retile inherits its charges rather than re-sampling", is
  **wrong and withdrawn**; see §8.2's round-4 correction. The retile re-samples
  through a full `BlobSampler` pass.) An 11 %-of-points charge correction moved
  Phase 1 by 4 terminals in 12131 and moved this track's tree by nothing at all.
- **Correcting the charge made the gate *stricter*, not looser** — 58.3 % →
  50.9 % below V, while the control half did not move at all. That is
  `calc_charge_wcp`'s `flag_p = (q > cut) || (q == 0)` (§1.2) doing exactly what
  it says: a zeroed plane *passes*. The bug had been buying those points a free
  pass on V, and removing it costs them one.

That second line is the one the redesign has been missing. §6(b) (a relative /
SNR floor instead of an absolute one) and §6(c) (combine the planes with a
consistency χ² instead of ANDing them) were argued in round 1 from doc 28's
**population** statistics. They are now supported by a **mechanism measured on
this event**: the three-plane AND treats "no charge" and "no readout" as the
same thing, and neither the absolute floor nor the AND has any way to tell them
apart. §6 is justified for this event on its own evidence.

**One downstream change, direction unknown, flagged for the owner.** In the ON
arm one cluster's STM verdict flips — 4 STM-tagged clusters become 3 (cluster 47
drops out; ours is 34, unaffected). Because PDVD runs
`nu_per_bundle_stm_only=true` (doc 25 §13.10), that removes one bundle's entire
per-bundle PR, and the calib dump follows: segments 23 → 2, vertices 27 → 3,
showers 4 → 0, `proj` 17 → 5. Tagger verdict count (82) and every warning line
are identical between arms, so the chain ran the same way; one verdict moved.
**Which verdict is right is not established here** — the OFF verdict was computed
on mutilated charge and the ON verdict on real charge, and deciding between them
needs a hand scan, not a count. It is recorded because it is the kind of change
that must not arrive unannounced with a default flip.

---

## 6. The redesign

The reframe first, because it is the whole argument:

> The stated philosophy is *"the charge is highest **locally**"*. The code
> implements *"above 4000 e globally **and** highest locally"*. The absolute
> term is the detector-dependent half, and it is the half that does not
> transfer.

Six elements, in descending confidence. Blast radius is deliberately confined to
`Steiner::Grapher::Config` and `calc_charge_wcp`'s estimator — **not**
`BlobSampler::stepped`, which every detector binds (SBND, uBooNE, PDHD) and
which doc 28 showed is not where the win is: the crossing ambiguity is
structural to PDVD's pitch, and the v7-uvwfit geometry made it slightly *worse*,
not better.

**(a) Aperture-matched charge — the owner's "combine nearby wires".** Replace
the single snapped wire with a charge integral over a window whose half-width is
a fixed **physical** distance, so the same configured number means the same thing
at 3 mm and 7.65 mm pitch. This is the element §7 measures.

The primitive already exists and is production-exercised:
`Grouping::get_ave_charge(point, apa, face, pind, radius)`
(`Facade_Grouping.cxx:640-670`, used by `NeutrinoVertexFinder.cxx:414,469` and,
summed over planes, `get_ave_3d_charge` in `NeutrinoShowerClustering.cxx:2945`).
Four caveats to settle before adopting it, none fatal:

- it returns the **mean** over in-radius cells, not the sum. A mean is roughly
  pitch-independent by construction, which is what a *relative* criterion wants;
  a sum is the dQ/dx-like quantity. §7 measures both — say which one the
  criterion uses rather than inheriting the choice by accident;
- the radius is Euclidean in (x_drift, y_pitch), so a circular aperture spans
  ~2.6× more slices than wires on PDVD's 2.96 mm × 7.65 mm cell. That may be
  desirable; it should be a stated choice;
- it re-fetches `local_pcs.at(ds_name).get("charge")` on **every call**,
  uncached, while the `kd2d` beside it is memoized. At ~160 k points × 3 planes
  that is ~0.5 M map lookups per cluster pass. The last Steiner-adjacent round
  was a 27-minute perf problem (doc 25 §13.11); hoist the array fetch, or use
  `Grouping::wire_charge_row(apa, face, plane, time_slice)`
  (`Facade_Grouping.cxx:1026-1035`), which returns a whole slice row;
- it returns `0.0` when the dataset is missing, which is indistinguishable from
  "no charge" — and §1.2 shows `calc_charge_wcp` then reads zero as *"don't hold
  it against the point"*. Given §5, that conflation is not hypothetical.

**(b) A relative floor, not no floor.** Dropping the absolute cut without a
replacement puts terminals on noise. `{u,v,w}charge_unc` is already stored per
point, so an SNR floor (`charge/unc`), or a floor set as a percentile of the
cluster's own charge distribution, is detector-independent by construction where
4000 e and 500 e are both calibration-coupled. This is what makes (a)
defensible rather than reckless, and it is what makes one configured number
correct on uBooNE, SBND and PDVD at once.

**(c) Combine the planes instead of ANDing them.** `calc_charge_wcp` ANDs three
fixed cuts and returns an RMS that *rewards* a single loud plane. With
`charge_unc` available, a χ² that the three planes see a **common** charge
penalises exactly the "losing" candidate doc 28 found PDVD manufactures 4× more
of — one plane's charge without the others' — and dead handling (`unc > 1e10`)
falls out of the weighting instead of `disable_dead_mix_cell`'s two ad-hoc
branches (whose divergence from the prototype is pr/29 D2, still open).

**(d) Neighbourhood in cm, not hops**, for the local-max step, removing coupling
#3 of §2. The cluster k-d tree makes this available at no new cost.

**(e) Listed, but explicitly not load-bearing: direction-aware normalisation.**
Dividing each plane's charge by the expected path length in that plane's pitch
cell, `Q_p / L_p(θ)`, is the principled way to make three planes comparable. It
needs a local track direction from PCA, which is density-dependent and
**undefined on the half of the track that has no points** — i.e. undefined
exactly where the symptom is. Keep it in the design, do not build on it.

**(f) A backstop, labelled as one.** A coverage guarantee — at least one terminal
per X cm along the cluster's principal axis, taking the local best even if it
misses the floor — would bound the failure mode directly. It is make-it-work,
not mechanism; it belongs last and only if (a)-(d) leave a residue.

### 6.1 The metrics to judge any of this by

Round 1 proposed **one** metric. The owner's round-4 brief states **three**
requirements, and they are not the same requirement: terminals must sit *on the
track or the vertex*, must not be *so many that the skeleton is smeared*, and
must not be *so few that a major feature of the event is missed*. One number
cannot guard three properties — a criterion that admits everything scores
perfectly on coverage.

| property required | metric | which failure it catches |
|---|---|---|
| **not too few** | largest terminal-free gap along the cluster's principal axis | 039349/14: **108.5 cm** |
| **not too many** | terminals per cm of skeleton | never measured on any detector |
| **on the track / vertex** | transverse distance from the **fitted** skeleton | never measured on any detector |

**Round 6 update.** All three were measured together for the first time in
§9.4, and the "not too many" row is the one that has since fired: the owner's
hand scan of the restored skeleton returned *too dense*. §10.3 root-causes that
— the local-max suppression is scoped to one blob, so terminal density is
candidate-bearing **blob** density (1.02 terminals per such blob, measured) —
and it is why round 7's lever is a cross-blob suppression pass rather than any
of §6's. The row below is therefore no longer "never measured".

The first is unchanged from round 1 and is still the headline: it is what was
actually wrong on 039349/14, it is detector-comparable, and it is computable
offline from the calib dump's `steiner`/`flag_terminal` arrays. The AND-gate
pass rate is a poor substitute — doc 28 moved it 17.4 % → 20.2 % with the v7
geometry and the symptom did not move at all.

The second and third are new, and they matter because every lever in §6 pushes
in the *same* direction: (a) an aperture, (b) a relative floor and (d) a metric
neighbourhood all admit **more** points. §7 already showed how far that can go —
at a relative floor of 0.2 the aperture admits 92.5 % of PDVD points and 100 % of
SBND's. Without a "too many" metric there is nothing to stop the redesign from
scoring a perfect gap by making every point a terminal.

A measured reference point for what "right" looks like, from the healthy half of
039349/14's track (§8.2's instrument, run on the same dump):

| half | terminals | skeleton length | density | perpendicular distance to the axis |
|---|---|---|---|---|
| above V (works) | 39 | 148 cm | **one per 3.8 cm** | median 1.68 cm, p90 2.48 cm |
| below V (starved) | 1 | 112 cm | one per 112 cm | — |

**One terminal per ~4 cm on a straight cosmic** is the only empirical anchor this
campaign has for the density axis; it is one track on one detector and should be
re-measured on SBND and uBooNE before it is treated as a target. PDVD has already
pulled the *"not too few"* lever once — `terminal_charge_threshold` 4000 → 500
(doc 25 §13.6) — without any instrument on the other two axes, which is exactly
the asymmetry this table exists to close.

The localization row above is bounded by the 3 cm corridor that selects it, so it
measures shape *inside* the corridor only. The real metric is distance to the
**fitted** skeleton, which is a PR product; building that instrument is named in
§10 as owed.

---

## 7. Feasibility: does combining nearby wires do what it should?

Measured on doc 28's PDVD event and its SBND control, since 039349/14 does not
exercise the estimator (§4). Aperture half-width **10 mm, physical**, converted
per detector and per plane from its own pitch and slice width: PDVD ±2 wires
(U/V/W) and ±4 slices of 2.96 mm; SBND ±4 wires and ±4 slices of 3.20 mm. Charge
is read from the `ctpc_*` maps. Candidacy uses the exact `calc_charge_wcp`
semantics; the aperture rows use a floor set as a fraction of that event's own
median estimate, which is element (b).

The single-wire lookup reproduces the stored `*charge_val` exactly — SBND
23182/23182 (U), 22840/22840 (V), 22009/22009 (W) — which validates the whole
2-D cross-reference before any conclusion rests on it.

| | PDVD 039252/2 (160930 pts) | SBND evt 16 (23930 pts) |
|---|---|---|
| median candidates per (U,V) crossing | **4** | **1** |
| crossings that are ambiguous | **83.6 %** | 21.0 % |
| points that are a losing candidate | **80.4 %** | 21.8 % |

reproducing doc 28 §4.3 exactly. Then, candidacy split by peak vs losing:

| estimator | floor [e] | PDVD cand / peak / losing / **ratio** | SBND cand / peak / losing / **ratio** |
|---|---|---|---|
| single wire (production) | 500 / 4000 | 0.611 / 0.750 / 0.577 / **0.77** | 0.241 / 0.259 / 0.176 / **0.68** |
| aperture SUM, rel 0.2 | 45141 / 46982 | 0.898 / 0.914 / 0.895 / **0.98** | 0.980 / 0.989 / 0.948 / **0.96** |
| aperture SUM, rel 0.5 | 112852 / 117455 | 0.741 / 0.793 / 0.729 / **0.92** | 0.845 / 0.858 / 0.798 / **0.93** |
| aperture MEAN, rel 0.2 | 1895 / 1349 | 0.925 / 0.923 / 0.925 / **1.00** | 1.000 / 1.000 / 0.999 / **1.00** |
| aperture MEAN, rel 0.5 | 4737 / 3371 | 0.768 / 0.793 / 0.762 / **0.96** | 0.930 / 0.934 / 0.917 / **0.98** |

`ratio` = losing/peak pass rate. **1.00 means a point's candidacy no longer
depends on whether it won its wire crossing.**

**The idea works, and for the stated reason.** A physical aperture removes almost
all of the peak-vs-losing asymmetry on **both** detectors (PDVD 0.77 → 0.92-1.00,
SBND 0.68 → 0.93-1.00), because both members of an ambiguous group sit inside
one aperture of the same charge. That is the mechanism doc 28 identified as
PDVD's dominant one, addressed directly rather than compensated for by lowering
a threshold.

Three honest qualifications:

1. **Removing the asymmetry is not the same as picking the right point.** Making
   the losing candidate pass at the peak's rate means candidacy stops
   *discriminating* on crossing ambiguity — it does not resolve the ambiguity.
   The discrimination has to come back from the local-max step (§6d) and from
   plane consistency (§6c), and neither is measured here. A criterion that
   admits everything is not obviously better than one that admits the wrong 60 %.
   This is the main open question for round 2.
2. **The floors are not tuned.** The relative floors here are illustrative; at
   rel 0.2 nearly everything passes on both detectors. The operating point has to
   be chosen against §6.1's gap metric, not against a pass rate.
3. **The aperture is a whole-event average here**, applied uniformly. It does not
   test the case §4 is about, and it cannot: on 039349/14 the candidates already
   exist.

---

## 8. Round 4: the owner's four questions

Round 4 changes **no behaviour**. Two toolkit changes, both inert in production:
one doctest, and three extra fields appended to the existing env-gated dump line
(`WCT_STEINER_PHASE_DUMP`, `getenv` null in production). No config key, so no
compiled-config diff and no A/B gate is owed — the reason is structural, not a
gate result: the probe cannot execute unless the variable is set.

The headline is **§8.2**, which was not one of the four questions: the terminal
starvation of §4.1 is now root-caused, and it is the *same* `AnodePlane` rule as
§5.2 at a *third* site — one the round-3 fix structurally could not reach. That
also explains round 3's null result, which §5.3 could only record.

### 8.1 Is the filter a bug? (question 1)

**Not on this event, and not as a port — with one per-detector exception and one
latent defect.** Four separate answers, because "the filter" is four things.

**(a) As the cause of the missing terminals: no, and this is now measured twice
over.** Round 2 measured the counts — Phase 2 removes 797 of 12131 (6.6 %),
Phase 3 removes **0**. Round 4 measures Phase 3's *operands*, which a count
cannot reach, over all 11334 terminals it tested event-wide:

| Phase 3 (`filter_by_path_constraints`) | n | fraction |
|---|---|---|
| `close_in_2d` (two of three planes within 1.8 cm) | 5695 | **0.503** |
| `dis_3d > 6 cm` | 774 | 0.068 |
| **both — i.e. removed** | **0** | **0.000** |
| of the 774 far-in-3-D terminals, 2nd-smallest 2-D distance < 1.8 cm | **0** | — |

Both halves of the test fire constantly; their **conjunction never does**. The
filter removes points that project close on two planes while sitting far away in
3-D — the signature of a wire-crossing ghost — and on this event not one terminal
has it. **Phase 3's zero is by design, not by breakage**, and that is now a
measurement rather than an assumption (cf. `feedback_zero_fires_not_dead_code`).

**(b) A latent defect that did not fire here, recorded so it is not
rediscovered.** `DynamicPointCloud::get_closest_2d_point_info` returns a raw
**−1.0** when the (plane, face, apa) 2-D tree is empty
(`DynamicPointCloud.cxx:375-377`), and Phase 3 tests `dis_2d < 1.8 cm`
(`SteinerGrapher.cxx:340-342`). **−1 reads as "very close"**, so a sentinel
inverts the guard's meaning and collapses the test into a bare `dis_3d > 6 cm`
removal. PDVD is precisely the detector whose clusters span several (apa, face)
volumes, so it is reachable — but it fired **0 of 11334 times** on this event.
Not a finding; a hazard with a measurement attached.

**(c) As a port, the answer is per-detector, and §3's version of it was wrong.**
The prototype always applies ±1 wire of slack and always tests slices t±1
(`PR3DCluster_steiner.h:285-290`, `:299-341`). The toolkit's defaults are
`wire_tol = 0` and a slice stride of 1 that, the map being tick-keyed, never
resolves. Both PDVD **and SBND** set both keys to the prototype values; **uBooNE
sets neither**, so uBooNE is the detector running a filter stricter than the
prototype it was ported from. See §3's round-4 correction and §8.4's table.

**(d) A design question that is not a bug.** Phase 2 tests the terminals of the
**retiled** cluster against the wire ranges of the **original** cluster's blobs
(`create_steiner_tree(src, …)`, `CreateSteinerGraph.cxx:276`, where `src` is the
pre-retile cluster). By construction that discards exactly the material the
retile exists to add. It is faithful to the prototype — `old_mcells` is the
original cluster's cell list there too (`PR3DCluster_steiner.h:238-249`) — so it
is not a port defect. Whether a *reference* filter should veto the *improvement*
it was run to obtain is a real design question, and it is the natural place to
look **after** §8.2 is fixed, not before: at 6.6 % it cannot be the binding
constraint while Phase 1 is emitting one terminal per 112 cm.

### 8.2 What actually starves Phase 1 — the retile's wire→channel mapping

This is the same `Gen::AnodePlane` rule as §5.2, at a **third** site, and this
one sits **upstream of Phase 1**.

**The invariant.** `AnodePlane::configure` pushes `plane_channels` back in wire
order while skipping every `segment() > 0` wire
(`gen/src/AnodePlane.cxx:242-256`). Therefore

```
channels[i]->ident() == wires[i]->channel()
```

holds **only** while no continuation has yet been skipped, and
`channels.size() == wires.size() − n_seg>0`.

**The site.** `ImproveCluster_1::make_iblobs_improved` — the retiler the Steiner
stage runs, `cm.improve_cluster_2` — indexes that vector by **wire index**:

```cpp
// clus/src/improvecluster_1.cxx:833-846
const auto& channels = wire_plane->channels();
for (size_t wire_idx = 0; wire_idx < plane_measures.size(); ++wire_idx) {
    if (plane_measures[wire_idx] > 0.0) {
        if (wire_idx < channels.size()) {         // past the end: DROPPED
            auto ichan = channels[wire_idx];      // after the first skip: WRONG channel
            slice_activity[ichan] = ...;
```

`plane_measures` is sized `total_wires = iplane->wires().size()`
(`aux/src/PlaneTools.cxx:55`) and is filled by wire index from the `ctpc_*` maps,
which are `wind`-keyed and correct (`improvecluster_1.cxx:436-441`,
`Facade_Grouping.cxx:889-926`). So the charge arrives correct and is then
**re-keyed onto the wrong channel**, after which `BlobSampler` stamps the retiled
points from that activity map. The identical line exists in base `RetileCluster`
(`retile_cluster.cxx:426-437`); no production pipeline reaches it (`cm.retile` is
commented out everywhere), so the live site is the `_improved` one.

**The geometry, from the toolkit's own `WireSchema::load` + `store.wires(plane)`
— the same accessor `AnodePlane` walks** (doctest
`clus/test/doctest_blob_sampler_wrapped_channel.cxx`, "round4" case; deliberately
**not** the raw `planes[].wires[]` JSON whose ordering forced §5.1's withdrawal):

| detector | planes | broken | `segment>0` wires | orphans (§5.2) |
|---|---|---|---|---|
| **PDVD** | 48 | **16** | **1568** | 1568 |
| **PDHD** | 24 | **16** | **11968** | 6400 |
| SBND | 6 | **0** | **0** | 0 |
| uBooNE | 3 | **0** | **0** | 0 |

Two things this table says that §5.2's could not. PDHD carries **11968**
continuations against only 6400 orphans — 5568 PDHD strips wrap back *inside*
their own plane, invisible to the orphan census yet still shifting
`plane_channels`. **It is this count, not the orphan count, that bounds the
retile defect.** And SBND/uBooNE are again immune structurally, not by gate.

Every broken PDVD plane is **287 wires with 98 continuations in one contiguous
band at one end** — never interleaved, which is what makes the consequence
predictable per plane instead of per wire. `first_bad` is 0 on 8 planes and
189 = 287 − 98 on the other 8:

| band position | `first_bad` | consequence for that plane |
|---|---|---|
| bottom | **0** | every wire index shifted by +98; the top 98 dropped ⇒ the plane is wrong **everywhere** |
| top | **189** | indices 0–188 correct; only the top 98 dropped |

**Anode 4 face 0 — where the flagship track lives — is U bottom (`first_bad=0`),
V top (`first_bad=189`).** Pinned in the doctest, because §8.2's whole argument
turns on it.

**The prediction, written before the measurement.** If this is the mechanism
then, in the *retiled* cluster: U is dead in **both** halves (whole plane
shifted); V survives above V but not below — §5.1 already measured that the
track's V wire is a continuation on **98.6 %** of points below the vertex and
**0.5 %** above, and in a `first_bad = 189` plane the continuations are exactly
the indices ≥ 189, i.e. precisely the band this code drops; W is untouched in
both halves (W has no `segment>0` wire, so
`channels[wire_idx]` is valid — the same internal negative control as round 3);
and therefore points below V hold **at most one** non-zero plane, which
`calc_charge_wcp`'s `ncharge > 1` rule (`Facade_Cluster.cxx:1105`) turns into
charge **0**, so they can never pass `charge > threshold` however much charge W
holds.

**The measurement** (`work/039349_14_d31r4dump`, the retiled cluster's own points,
read out of the extended dump):

| retiled cluster (`P0`) | n | U ≠ 0 | V ≠ 0 | W ≠ 0 | **≥2 planes ≠ 0** |
|---|---|---|---|---|---|
| **below V** (starved) | 1239 | **0.000** | **0.004** | 0.588 | **0.002** |
| **above V** (control) | 631 | **0.000** | **0.349** | 0.434 | **0.217** |

Every clause of the prediction holds. Below V, **99.8 % of the points cannot be
candidates at any threshold** — not because the charge is missing from the
detector, but because the retile re-keyed it.

The counts close, which is a stronger tie-off than the fractions: 0.002 × 1239 ≈
**2–3 points** below V can reach the gate at all, and Phase 1 — which suppresses
non-maximal candidates within each blob (§1.3) — returns exactly **1** terminal.
Above V, 0.217 × 631 ≈ **137** reach the gate and Phase 1 returns **44**. The
44-vs-1 asymmetry that opened this document is fully accounted for by one line of
channel bookkeeping.

Cross-checks that keep this from being a coincidence:

- The charge really is there. Doc 31 §5.3 measured the *input* cloud with the
  round-3 knob on: below V, V matches `ctpc_a4f0pV` exactly on 677/721 and U on
  709/721. The retile loses charge the sampler had already got right.
- W behaves exactly as a never-wrapped plane must: non-zero in both halves,
  0.588 / 0.434, and it is the only plane surviving below V.
- U is 0.000 on **all 1870 corridor points** but 0.147 over the whole 5705-point
  retiled cluster — the rest of the cluster lies on (apa, face) volumes whose U
  plane is not broken.

**This is why round 3's fix could not have worked, and §5.3's stated reason was
wrong.** §5.3 wrote that the retile "inherits its charges from the input point
cloud rather than re-sampling". **It does not.** `ImproveCluster_1::mutate` runs
a full `BlobSampler` pass over the retiled blobs (`Aux::sample_live`,
`improvecluster_1.cxx:148`), which does consult `p_chi2i`
(`BlobSampler.cxx:383`). §8.2's own measurement is the proof: if the charges were
inherited, the knob-ON retiled cluster would show the real V charge below V — the
input cloud matches `ctpc_a4f0pV` on 677/721 there — and it shows 0.004.

The actual reason the knob is inert inside the retile is narrower and worth
stating exactly. `slice_activity`'s **keys are drawn from `channels` itself**
(`improvecluster_1.cxx:840`), so the map can only ever contain segment-0-listed
channels. The knob changes behaviour only on a `p_chi2i` **miss**, i.e. on an
orphan wire — and there knob-ON searches an ident map that by construction cannot
hold that orphan, while knob-OFF reads `channels[0]`'s entry, which was absent on
this event. Both give 0, hence byte-identical. Note that this last step is
**empirical, not structural**: `channels[0]` could carry activity in another
event, in which case the two arms would differ (and the legacy arm would be
importing a foreign channel's charge, §5.2's "wrong non-zero" failure mode).

So the conclusion "set the knob at clustering time" is unchanged and still
verified; only the mechanism attributed to it was wrong. The round-3 fix remains
correct and worth having — it repairs the persisted point cloud that clustering,
Q/L and the taggers read — it simply cannot reach the Steiner stage, because the
retile discards those charges and re-derives them through a map that never had
the wrapped channels in it.

**Blast radius: PDVD only.** SBND and uBooNE have no continuations. PDHD has more
than PDVD but **does not run the Steiner stage at all** — the strings
`CreateSteinerGraph`, `cm.steiner` and `steiner` do not appear in any of the
37 `cfg/pgrapher/experiment/pdhd/*.jsonnet` or the 10
`wcp-porting-img/pdhd/*.jsonnet` files (the only hit in either tree is a
`pdhd/docs/*.md`), and no PDHD entry point imports a `pr.jsonnet`. That is the
opposite of §5.2's
conclusion for the *sampler* bug, where PDHD was the worse-hit detector, and the
difference is worth stating plainly so the two are not conflated.

**Not established here:** how much of the *rest* of PDVD's Steiner output this
moves. One track on one anode-face is measured. A fix and a manifest-scale
re-measurement are round 5 (§9), which ran and closed the gap.

### 8.3 Does the fix affect event 298595? (question 2)

**No — cluster 86 is bit-identical between the two arms.** But the event around
it is not, so both halves of that answer are worth having.

Arms `work/039252_2_d31r4{off,on}`, run 039252 event 2, same imaging tarballs
hardlinked from `039252_2_d27fresh`, one knob apart, knob set at **clustering**
time, both on the same pinned library.

*The knob does bite on this event* — `P(charge_val == 0 | segment == 1)`, from
the two output point clouds:

| plane | segment | n points | OFF | ON |
|---|---|---|---|---|
| U | 1 | 21889 | 0.349 | **0.097** |
| V | 1 | 18886 | **0.829** | **0.051** |
| W | 0 (never wrapped) | 196745 | 0.052 | **0.052** |

Deterministic groups (`P == 1.000`) go **3 → 0**. W does not move: the control.

*Doc 30's symptom does not move at all:*

| doc 30's instrument, cluster 86 | OFF | ON |
|---|---|---|
| `TaggerCheckSTM: cluster 86 → STM=` | **1** | **1** |
| `form_map_graph` zero-quantity drop, segment `gi=2` | **108 → 2** | **108 → 2** |
| pre-drop shape | chord 64.18 cm, `max_perp_dev` **0.000 cm** | chord 64.18 cm, **0.000 cm** |
| `track_fit-global` points on cluster 86 | 41 | 41 |
| `stm_fit-global` points on cluster 86 | 141 | 141 |
| cluster-86 coordinates, all three layers | — | **identical** |

**This confirms doc 30 round 3 rather than competing with it.** Doc 30 traced the
lost long leg to `fit_exclusion` contention with a duplicated segment placed over
the same charge by a mid-fit `do_rough_path` edit, and explicitly *not* to charge
attribution. A charge fix that takes this event's segment-1 V zeros from
**82.9 % to 5.1 %** and still leaves the leg at 2 points is a direct test of that
attribution, and it passes. Doc 30's recommendation 1 — fix the duplicate
segment — stands unchanged.

*The rest of the event does move, and it is the same class of downstream change
round 3 flagged:*

| | OFF | ON |
|---|---|---|
| clusters with STM = 1 | 6 (39, 40, 55, 86, 87, 90) | **7** (+ **109**) |
| `save_stm_fit` segments stored | 24 | 22 |
| `track_fit-global` points / clusters | 412 / 5 | **478 / 9** |
| `vertices-global` points / clusters | 29 / 5 | **40 / 9** |
| calib-pr candidates / vertices / showers | 5 / 7 / 1 | **6 / 10 / 3** |
| `mabc-pr.zip` members differing (of 22) | — | **6** |

One cluster (109) is newly STM-tagged, and because PDVD runs
`nu_per_bundle_stm_only=true` (doc 25 §13.10) that admits a bundle to per-bundle
PR that was previously skipped — the same coupling as round 3's flip, in the
opposite direction (round 3 lost a tag; here one is gained). **Which verdict is
right is not established here** and needs a hand scan, not a count: OFF was
computed on mutilated charge and ON on corrected charge. Recorded for the same
reason as §5.3's: it must not arrive unannounced with a default flip.

*Caveat on the absolute numbers.* This is a shared tree and a peer's uncommitted
`TrackFitting` work was in `local/lib` when the snapshot was pinned
(md5 `fe1364fd75d244819112402605c56cdb`). Both arms ran on that one snapshot, so
the OFF↔ON comparison above is clean; the absolute counts are not guaranteed to
match doc 30's published arms, and the cluster-86 rows that do match
(41 / 141 / 108→2 / 0.000 cm) are the evidence that the comparison is on the same
object doc 30 studied.

### 8.4 The terminal-selection threshold, per detector (question 4)

| knob | C++ default | **uBooNE** | **SBND** | **PDVD** | **PDHD** |
|---|---|---|---|---|---|
| `terminal_charge_threshold` | **4000 e** (`SteinerGrapher.h:92`) | absent ⇒ **4000** | absent ⇒ **4000** | **500** (`pdvd/wct-pr-perevt.jsonnet:678`) | — |
| `terminal_wire_tol` | 0 (`SteinerGrapher.h:53`) | absent ⇒ **0** | **1** (`sbnd/wct-pr-perevt.jsonnet:381`) | **1** (`:380`) | — |
| `terminal_adjacent_slice` | false (`SteinerGrapher.h:62`) | absent ⇒ **false** | **true** (`:382`) | **true** (`:381`) | — |
| `edge_charge_forward_dead_mix` | false (`SteinerGrapher.h:81`) | absent ⇒ **false** | **true** (`:397`) | **true** (`:396`) | — |
| wire pitch U / V / W | — | 3.00 / 3.00 / 3.00 mm | 3.00 mm | **7.65 / 7.65 / 5.10 mm** | 4.67 / 4.67 / 4.79 mm |

**PDHD has no column: it does not run the Steiner stage** (no
`CreateSteinerGraph` in any of its 54 config files). The prototype's own value is
**4000 e** (`PR3DCluster_steiner.h:759`, and `calc_charge_wcp`'s default
`charge_cut = 4000` at `PR3DCluster.h:129`), so uBooNE and SBND both run the
prototype number and only PDVD has moved — 4000 → 500 (doc 25 §13.6: no-steiner
exits 39 → 14, STM tags 16 → 18, the first Bragg-clean track appears at 500).

Read the threshold against the pitch row: **4000 e at 3.00 mm on uBooNE, 500 e at
7.65 mm on PDVD.** Per unit length that is 13333 e/cm versus 654 e/cm — a factor
20 apart on a quantity that ought to be comparable, which is §2's coupling #1
stated as a number.

Two findings that belong with the table:

- **PDVD's 500 never reaches the retiler.** `ImproveCluster_2` default-constructs
  its `Steiner::Grapher::Config` (`improvecluster_2.cxx:91-96`, only `dv`,
  `pcts`, `perf` are set) and then runs the full terminal finder **twice**, via
  `establish_same_blob_steiner_edges` at `:107` and `:158`
  (→ `SteinerGrapher.cxx:723`). So inside the retiler, terminals are selected at
  the C++ default **4000 e** with `wire_tol = 0`, on every detector, unreachable
  from jsonnet. Those terminals set the intra-blob 0.8 / 0.9 edge weights of
  `basic_pid` and `ctpc_ref_pid`, whose shortest paths become
  `orig_path_point_indices` / `temp_path_point_indices` and feed
  `hack_activity_improved` (`improvecluster_2.cxx:113`, `:164`, `:194`, `:199`).
  They therefore shape **which blobs the retile creates at all** — upstream of
  Phase 1, not merely upstream of Phase 3. (The path Phase 3 actually filters
  against comes from `CreateSteinerGraph.cxx:266`, on a graph whose same-blob
  edges were established at `:252` by the `sg` grapher, which *does* carry PDVD's
  500.) PDVD is therefore running **two different terminal thresholds in one
  stage**, the earlier and more consequential of which is not configurable.
- **The other hard-coded 4000 is inert** — `calculate_vertex_charges`
  (`SteinerGrapher.cxx:936`, called with the literal at `:1146`) passes
  `charge_cut` into `calc_charge_wcp` but keeps only `.second`, and `charge_cut`
  governs only the discarded flag. Checked, harmless, recorded so it is not
  re-flagged as a PDVD inconsistency.

### 8.5 Would you recommend an improvement? (question 3)

**Yes, but not first, and the order has changed because of §8.2.**

1. **Fix the retile's wire→channel mapping.** It is a bug, not a design choice;
   it is upstream of every criterion §6 would change; it is PDVD-only; and it is
   small — resolve the channel from `wire_plane->wires()[wire_idx]->channel()`
   through the anode's ident map instead of indexing `channels` positionally,
   at `improvecluster_1.cxx:840` and its twin at `retile_cluster.cxx:433`.
   Same shape as round 3: default-OFF knob, SBND/uBooNE structurally unaffected,
   the doctest above as the premise, §6.1's three metrics as the acceptance test.
   One design point already settled so round 5 need not rediscover it: two wires
   of the *same* plane can share a channel (5568 such strips on PDHD), so the
   corrected lookup can write the same key twice in one slice. That is benign —
   `PointTreeBuilding::add_ctpc` walks activity forward and writes the *same*
   channel charge into both wires' ctpc rows, so the two writes agree by
   construction. And on SBND/uBooNE wire index and channel-list index coincide
   exactly, so the corrected code is byte-identical there **by construction**
   rather than by gate.
2. **Re-measure before designing.** The redesign's entire case on *this event* is
   the 108.5 cm gap. §8.2 says Phase 1 was fed a plane of zeros and one plane of
   truth over that stretch, and `ncharge > 1` then guarantees no candidate at any
   threshold. **Until step 1 lands, this event cannot tell us whether the
   criterion is wrong** — it only tells us the criterion was starved. Round 2 made
   the mirror-image mistake in the other direction and §4.2 records the cost.
3. **Then judge §6 on the evidence that survives.** Doc 28's population evidence
   and §7's aperture result are untouched by §8.2 — they were measured on the
   *input* point cloud, not the retile — so the case for (b) a relative/SNR floor
   and (c) a plane-consistency χ² instead of the three-plane AND stands on its
   own. §5.3's mechanism argument stands too: `(q > cut) || (q == 0)` cannot tell
   "no charge" from "no readout", and §8.2 is now the third distinct way that
   conflation has produced a silent wrong answer in this campaign.

On the owner's stated requirement specifically: §6.1 now carries all three
properties, and the honest position is that **only one of the three is
instrumented today**. Every lever in §6 pushes toward *more* terminals, and §7
already showed a floor at rel 0.2 admitting 92.5 % of PDVD points. Building the
density and localization metrics is a prerequisite for the redesign, not a
follow-up to it — otherwise the redesign will be graded by a metric it cannot
fail.

---

## 9. Round 5: the retile mapping is fixed, and the 108.5 cm gap closes to 1.8 cm

§8.5 recommended fixing the retile's wire→channel mapping first and
re-measuring before touching §6. Done, and the result is unambiguous.

### 9.1 The fix

`ImproveCluster_1::make_iblobs_improved` (`improvecluster_1.cxx:840`) and its
unreached twin in base `RetileCluster` (`retile_cluster.cxx:433`) now resolve a
wire's channel by **ident**, through `IAnodePlane::channel()`, instead of
indexing `IWirePlane::channels()` positionally:

```cpp
if (ianode) {                                     // knob ON
    if (wire_idx < pwires.size()) {
        ichan = ianode->channel(pwires[wire_idx]->channel());
    }
}
else if (wire_idx < channels.size()) {            // legacy, reproduced exactly
    ichan = channels[wire_idx];
}
if (!ichan) continue;
```

**Knob:** `wrapped_channel_activity`, on `RetileCluster` (so `ImproveCluster_1`
and `_2` inherit it), **C++ default false**, threaded to PDVD through
`cm.improve_cluster_2` with the key-suppression idiom and the runner TLA
`retile_wrapped_channel_activity`.

Its precondition is pinned by a doctest rather than assumed: `IAnodePlane`'s
ident map is filled only from segment-0 wires, so the lookup resolves an orphan
of one plane only because that channel's segment-0 wire lives in **another plane
of the same anode**. Measured on all four production geometries — **0**
channels break that assumption anywhere.

### 9.2 Gates

| gate | result |
|---|---|
| `./build/clus/wcdoctest-clus` | **276/276** (2 new cases; 274 before) |
| compiled PDVD PR config, knob OFF vs `HEAD` | **byte-identical**, 278613 B |
| compiled PDVD PR config, knob ON | differs by **exactly one key on one node** (the `ImproveCluster_2` retiler) |
| compiled **SBND** PR config (shared `common/clus.jsonnet` was edited) | **byte-identical**, 253993 B, key absent |
| compiled **uBooNE** config | **byte-identical**, 255717 B |
| PDVD 039349/14 end-to-end, knob OFF vs round 4 | `calib-pr` **byte-identical**, 7510870 B |
| freshness | source 16:01:20 → `local/lib` 16:01:51, md5 `26b6cdc3481ae7d67a6f058f04ad8e53` |

The two shared-detector rows matter because this round edited
`cfg/pgrapher/common/clus.jsonnet`, which SBND and uBooNE both import. They are
byte-identical *and* structurally immune (the "round4" doctest pins zero
`segment>0` wires on either geometry), which is the stronger of the two.

### 9.3 The measurement, and the composition finding

Three PR-only arms on the same production input point cloud
(`work/039349_14_d31r5{off,on,both}`); the retile lives in the PR stage and
rebuilds its activity from the ctpc clouds, which are correct independently of
round 3's knob, so this is a clean single-knob change from production.

The retiled cluster's own per-plane charge along the starved stretch — the
quantity Phase 1's gate reads:

| below V, retiled `P0` | n | U ≠ 0 | V ≠ 0 | W ≠ 0 | **≥2 planes** |
|---|---|---|---|---|---|
| production (both knobs off) | 1239 | 0.000 | 0.004 | 0.588 | **0.002** |
| `retile_wrapped_channel_activity` only | 1102 | **0.495** | 0.009 | 0.676 | **0.378** |
| **both knobs** | 1100 | **0.503** | **0.665** | 0.674 | **0.579** |

**The two fixes are complementary, and the split is exactly what the geometry
predicts.** On a4f0, U's 98 continuations sit at wire indices 0–97 and V's at
189–286 (§8.2's table). The retile fix alone recovers **U**, because U's shifted
wires are *segment-0* wires whose channels the plane does list — `BlobSampler`'s
`p_chi2i` finds them. It cannot recover **V**, because V's continuations are
*orphans*: the activity map now holds them, but the sampler's own lookup still
misses, and round 3's `wrapped_channel_charge` is what repairs that. Hence
V: 0.004 → 0.009 → **0.665**.

This retroactively justifies round 3. §5.3 could only say the fix was correct
but ineffective on this stage; it is in fact **necessary** here, and was merely
blocked by a second defect downstream of it. The premise is verified rather than
assumed: `wrapped_channel_charge` is **absent from the `on` arm's compiled
config** (0 occurrences, against 16 in the `both` arm), so the U-only recovery
really is this knob acting alone.

**Was the legacy path losing charge, or importing someone else's?** §5.2 warned
that the failure mode admits a *wrong non-zero* value, and on a `first_bad = 0`
plane the legacy code wrote each measure onto the channel of a wire 98 positions
away — a real, listed channel of that plane, which the sampler then resolves. On
the one plane this event lets us identify, a4f0's U (a `first_bad = 0` plane),
the OFF arm has U ≠ 0 on **0 of 1870** corridor points: every shifted lookup
landed on a channel with no activity in that slice. **On this event the failure
was pure loss.** The import mode remains possible by construction — it is not
demonstrated here, and the dump carries no (apa, face) so the rest of the event
cannot be split the same way.

**And the symptom that opened this campaign:**

| below V (111.5 cm), from the calib dump's `steiner` section | OFF | retile fix | both |
|---|---|---|---|
| steiner points | 5 | **590** | **666** |
| terminals | 1 | **174** | **198** |
| **largest steiner-free gap** | **108.5 cm** | **3.0 cm** | **1.8 cm** |
| control half above V, same gap | 65.6 cm | 59.5 cm | 59.5 cm |

**The 108.5 cm gap is closed.** Doc 26 §7.5's observation — a continuous straight
cosmic whose Steiner cloud covers only the half above the vertex — was a
channel-indexing bug in the retiler, not a property of the terminal criterion.

Two honest riders:

- **The control half improves too** (65.6 → 59.5 cm, 258 → 330 steiner points),
  because a4f0's U plane was mis-keyed there as well. So the "control" was never
  clean for U; it was clean for V, which is why V's numbers above V barely move
  (0.349 → 0.357). Everything V does below V is the signal.
- **The retile itself changes**, as §8.4 said it would: `create_steiner_tree`
  calls go 50 → 47 → 46 and the retiled cluster 5705 → 5132 → 5139 points,
  because `ImproveCluster_2`'s internal terminal finding shapes which blobs the
  retile creates. The comparison is therefore between two slightly different
  retiled objects, not two labellings of one — which is why the gap metric,
  computed on each arm's own geometry, is the right judge.

### 9.4 What §6.1's new metrics now say

This is the first time all three of the owner's properties can be read at once:

| | production | retile fix | both | control half (both) |
|---|---|---|---|---|
| **not too few** — largest gap | 108.5 cm | 3.0 cm | **1.8 cm** | 59.5 cm |
| **not too many** — one terminal per | 112 cm | 0.64 cm | **0.56 cm** | 2.5 cm |
| **on the track** — median ⊥ distance | (n=1) 2.09 cm | 0.97 cm | — | 1.72 cm |

Coverage and localization are both good: the restored terminals sit *closer* to
the track axis (median 0.97 cm) than the control half's do (1.72 cm).

**The density row is the one to look at, and it is a genuine question for the
owner, not a defect I am reporting.** Below V the fix produces one terminal every
**0.56 cm**, against **2.5 cm** on the control half.

Before reading 4.4× into that, normalise it — the two halves do not have the same
point density either:

| both-knobs arm | below V | above V | ratio |
|---|---|---|---|
| retiled points per cm | 9.87 | 4.19 | 2.4× |
| terminals per cm | 1.78 | 0.40 | **4.4×** |
| terminals **per point** | **0.180** | **0.096** | **1.9×** |

So most of the 4.4× is just that the starved stretch has more blobs per cm; the
criterion's own behaviour differs by about **2×**, not 4.4×. That is a much
milder statement, and it is the one to hand the owner.

Whether one terminal per 0.56 cm is right — this stretch really does carry dense,
continuous charge — or is the *"smeared, hard to see the skeleton"* regime the
brief warns about, is a judgement a count cannot make. **§6.1's density metric
existed for exactly one round before it had something to say.** Deciding it needs
a Bee hand-scan of the restored skeleton, and that is round 6's first item.

---

## 10. Round 6: the owner's hand scan, and the three decisions it settled

Round 5 ended with three questions that no count could answer, and one that was
purely a design call. All four are now answered. This section records the scan,
the mechanism behind its one negative verdict, and what shipped as a result.

### 10.1 The scan

Three Bee sets were built from the existing arms — no new reconstruction — and
handed over for a hand scan. Set A carries **two layers that had never been in a
Bee zip before**, `steiner-global` and `steinerterm-global`, built from the calib
dump's `steiner` section (per-cluster `x/y/z` + `flag_terminal`), which is the
same source §9.4's numbers come from.

| set | Bee UUID | contents | question |
|---|---|---|---|
| **A** density | `f4edc748-91ef-4dce-a67f-56a8c4fd5b63` | 039349/14, 3 events: 0 production, 1 retile fix, 2 both knobs | is one terminal per 0.56 cm readable, or smeared? |
| **B** cluster 47 | `9c700983-1a2f-41a3-a4e6-f4ea182d5152` | 039349/14, round-3 knob OFF (0) vs ON (1) | is cl47 an STM, and is it one particle? |
| **C** cluster 109 | `85529727-bd67-4189-8817-8e927e40b507` | 039252/2, round-3 knob OFF (0) vs ON (1) | does cl109 stop inside the volume? |

The frame was proved before the sets were shipped rather than assumed: cluster
34's 2561 steiner points nearest-neighbour into that arm's own Bee
`clustering-global` at a **median 0.83 cm** and land on Bee `cluster_id` 34 on
**2561 of 2561**, so the new layers sit on the charge and share its numbering.
Builder: `scripts/build_bee_sets.py`.

**The owner's verdicts, in their words:**

- **Set A** — "the steiner-global are quite good, in terms of continuous, but
  the steiner terminal seems to be too dense; I recall that we have some local
  maximum selection right?"
- **Set B** — "this track looks like a TGM, not a STM, despite it has some gaps."
- **Set C** — "the 109 could be a STM."

### 10.2 What B and C settle: both downstream verdict changes point the same way

§5.3 and §8.3 each recorded an STM verdict that moved when round 3's knob went
on, and each said in terms that **which verdict is right was not established**.
The hand scan establishes both, and both go the fix's way:

| | object | knob OFF | knob ON | owner's read | so the ON verdict is |
|---|---|---|---|---|---|
| §5.3 | cl 47, 039349/14 | STM-tagged | **not** tagged | **TGM**, not STM | **correct** |
| §8.3 | cl 109, 039252/2 | not tagged | **tagged** | could be an STM | **plausible** |

Cluster 47's geometry supports the reading independently of the tag: it is
**760 cm** end to end and runs x −343 → +343, i.e. anode to anode across *both*
drift volumes. A stopping-muon verdict on that object was the anomaly; removing
it is the correction. This is the evidence §5.3's "must not arrive unannounced
with a default flip" was asking for, and it is what unblocks §2 below.

### 10.3 Set A's negative verdict, and why the peak finder cannot fix it

**Yes, there is a local-maximum selection — and it is scoped to one blob, which
is exactly why it cannot thin these terminals.**

`find_peak_point_indices` (`SteinerGrapher.cxx:493`) is a genuine local-max
suppression: candidates are walked in order of decreasing charge, each takes an
`nlevel = 1` BFS neighbourhood on the graph, a candidate is demoted if any
neighbour carries more charge, and finally connected components of surviving
peaks are collapsed to the one point nearest each component's centre of mass.

But `find_steiner_terminals` (`SteinerGrapher.cxx:681`) calls it **once per
blob**, over that blob's points alone. The nlevel hops are taken on the full
graph, yet the candidate set never leaves the blob, so the suppression can thin a
blob to one terminal and can **never remove a blob's last one**. Terminal count
is floored at the number of candidate-bearing blobs.

Measured, not inferred — a new env-gated counter (`steiner_p1_blobs`, under the
existing `WCT_STEINER_PHASE_DUMP` gate) prints `nblob / ncand_blob / nterm` per
call, over the whole of 039349/14:

| component | calls | blobs | candidate-bearing | terminals | terminals per candidate-bearing blob | `nterm == ncand` |
|---|---|---|---|---|---|---|
| `CreateSteinerGraph` | 92 | 69522 | 28726 | 29342 | **1.021** | 42/92 calls exactly |
| `ImproveCluster_1` (inside `_2`) | 92 | 49376 | 15120 | 15190 | **1.005** | 73/92 calls exactly |

The floor is essentially saturated: the peak finder is already emitting about one
terminal per candidate-bearing blob, and the excess over 1.000 is blobs whose
candidates fall into more than one connected component. **So terminal density is
blob density**, and §9.4's 4.4× is the 2.4× retiled-blob-density difference plus
that small excess — which is the same decomposition §9.4 arrived at from the
other side (1.9× per point).

Two consequences worth stating plainly:

- **No threshold and no `nlevel` can thin this.** Raising
  `terminal_charge_threshold` removes candidate-bearing blobs entirely (and with
  them coverage, which is the metric round 5 just fixed); raising `nlevel`
  widens a neighbourhood that is intersected with a single blob's points anyway.
  The lever has to be a suppression pass that sees **across** blobs.
- **Per-blob is prototype-faithful** (`find_steiner_terminals` mirrors the
  prototype's per-mcell loop), so a cross-blob pass is a deliberate divergence
  under M15, not a bug fix. It needs its own default-OFF knob and its own gate.
  That is round 7, and §6.1's density metric is now the thing that grades it.

### 10.4 What shipped

Four owner decisions, three of them code.

**(1) Q2 — both knobs are PDVD production.** `wrapped_channel_charge` (round 3)
and `retile_wrapped_channel_activity` (round 5) now default **true** in PDVD's
configs: `protodunevd/clus.jsonnet`, `protodunevd/pr.jsonnet`, and the runners
`pdvd/wct-clustering.jsonnet` and `pdvd/wct-pr-perevt.jsonnet`. **The C++
defaults are unchanged (both still `false`)**, so PDHD — which carries 2.5× the
orphan fraction and still has no event-level gate here — and every other detector
are untouched. This is a behaviour change by intent; §10.5's gate proves only
that it *is* a default change and nothing more.

**(2) Q5 — the Steiner stage no longer runs two terminal thresholds.**
`ImproveCluster_2` gains a `terminal_charge_threshold` key (C++ default 4000, the
historical value) which its internal `Grapher::Config` now carries, and PDVD sets
it from the same `steiner_terminal_charge` value `CreateSteinerGraph` uses — one
number, so the two cannot drift apart again. §8.4 recorded the inconsistency;
this closes it.

**(3) Q6 — `traj_degenerate_wcpts_fallback` is retired.** Removed from
`TrackFitting::Parameters`, both `set_parameter`/`get_parameter` cases,
`TaggerCheckNeutrino`'s member and config round-trip, and the PDVD runner TLA.
The surviving expression is the legacy `segment->fits().empty()`. Retirement is
byte-identical **by construction**: the default was 0 and no detector's config
ever set it. Its doctest is replaced by a guard that the name now *throws* from
`get_parameter`/`set_parameter`, so a silent reintroduction fails the suite.

**(4) Q4 — §6 is parked, not retired.** The redesign's case on this event is
gone; what survives untouched is doc 28's population evidence and §7's aperture
result, both measured on the input point cloud. Either would reopen it.

### 10.5 Gates

| gate | result |
|---|---|
| `./build/clus/wcdoctest-clus` | **277/277**, 4006 assertions (1 new case: the `ImproveCluster_2` default) |
| PDVD PR config: new defaults, sync suppressed, **vs HEAD + explicit knob TLAs** | **byte-identical**, 279974 B |
| PDVD PR config: production default vs that | **exactly one key on one node** (`terminal_charge_threshold` on `ImproveCluster_2`), 59 nodes compared |
| PDVD clustering config: new default vs HEAD + explicit TLA | **byte-identical**, 198097 B |
| **SBND** `wct-pr-perevt` (shared `common/clus.jsonnet` edited) | **byte-identical**, 253993 B |
| **uBooNE** `uboone-mabc` | **byte-identical**, 255717 B |
| end-to-end: `d31r6prod` (new defaults) vs round 5's `d31r5both` | `calib-pr` **byte-identical**, 7160347 B |
| freshness | source 17:30:23 → `local/lib` 17:34:01, md5 `be0aa5a7d88bc4bf046425692cadbde6` |

The first row of the config block is the one that carries the argument: it
compiles the *new* tree with the sync suppressed against the *HEAD* tree with
both knobs passed as explicit TLAs, and gets the same bytes. That proves the
commit changed defaults and plumbing and nothing else. It is deliberately **not**
an end-to-end byte-identity claim — the flip changes reconstruction output, which
is the point of it.

### 10.6 What the Q5 sync actually costs

Measured before it was set, on a fourth arm (`d31r6sync`, the sync applied by
TLA), because lowering the retiler's threshold admits more terminals into
`hack_activity_improved` and could plausibly have made the density the owner just
flagged *worse*. It does not:

| 039349/14, whole event | prod (4000 inside `_2`) | sync (500) |
|---|---|---|
| `ImproveCluster_1` candidate-bearing blobs | 15120 | **31360** (2.07×) |
| steiner points / terminals, all clusters | 45874 / 13427 | 45935 / **13431** |
| clusters whose steiner cloud changes | — | **1** (cluster 44: +61 points, +4 terminals) |
| segments / vertices / showers / candidates | 20 / 23 / 2 / 3 | **identical** |
| **cluster 34 below V**: points / terminals / gap | 666 / 198 / 1.8 cm | **666 / 198 / 1.8 cm** |

So the retiler's internal terminal count doubles, and the flagship track's final
Steiner cloud is bit-identical. One unrelated cluster moves. The sync removes a
real inconsistency at close to zero cost **on this event** — one event is not a
manifest, and it inherits the same validation debt as the two flips.

### 10.7 Round 7

1. **The cross-blob suppression pass** §10.3 identifies — a second peak stage
   that sees beyond one blob, behind a default-OFF knob, graded by §6.1's three
   metrics together (a lever that improves density while destroying coverage is
   not an improvement). This is the first item in this campaign that the *"not
   too many"* half of the owner's brief has actually asked for.
2. **Manifest-scale validation of everything round 6 flipped.** Three changes
   now ship ON for PDVD on the evidence of two events and two hand-scanned
   clusters. That was enough to flip; it is not enough to leave unvalidated.
3. **PDHD still has no event-level gate** for `wrapped_channel_charge`, and it
   is the detector with 2.5× the orphan fraction. One knob-OFF PDHD event closes
   it cheaply, and PDHD runs no Steiner stage so round 5's knob cannot reach it.

---

## 11. Scope of each round, and what round 7 is

**Round 1** (doc + scripts, no code): the algorithm review, the four uBooNE
couplings, the §6 redesign, §7's aperture feasibility. Its §4 conclusion was
overturned by round 2 — see §4.2.

**Round 2** (`b74f60ed`): one toolkit change, an **env-gated, log-only** terminal
dump in `create_steiner_tree` (`WCT_STEINER_PHASE_DUMP`, default OFF). No config
change, no A/B gate owed. It produced §4.1's census, which named Phase 1.

**Round 3** (`4e2bd2f1`): the wrapped-segment charge bug of §5 is **root-caused
(§5.2), fixed behind a default-OFF knob, and gated (§5.3)** — and the fix is
**measured not to cure the symptom**. Gates in §5.3's table. What round 3
establishes, in order of how much it should change anyone's plans:

1. **The redesign is still needed, and now for a reason measured on this event.**
   Correct charges leave the terminal-free gap at 108.5 cm, unchanged to the
   digit. §6(b) and §6(c) no longer rest on doc 28's population evidence alone:
   `calc_charge_wcp`'s `(q > cut) || (q == 0)` cannot distinguish "no charge"
   from "no readout", which is why removing the zeros made the gate *stricter*
   (58.3 % → 50.9 %) rather than looser.
2. **A real bug is fixed, and it is not PDVD's alone.** PDHD carries 2.5× the
   orphan fraction and is protected only by the C++ default.
3. **A one-verdict downstream change exists** and needs a hand scan before any
   default flip (§5.3, last paragraph).

Still not done, and explicitly owed:

- **Turning the knob on anywhere by default is an owner call**, not this round's.
  It is a behaviour change on PDVD and PDHD; §5.3's evidence says it is *correct*
  but says nothing about whether the tuning downstream of it still holds. The
  minimum before a flip: the STM verdict on cluster 47 hand-scanned, and a
  knob-ON pass over a manifest rather than one event.
- **PDHD has no event-level gate here** — only its wires file was examined. One
  knob-OFF PDHD event would close that, cheaply.
- **The dead-blob sampler is untouched** (§5.3): same bug, different consumer,
  its own gate.
- **Anodes 0-3's `wire_index` ↔ v7 geometry mismatch** (§5.1 correction) is
  unexplained. W matches on all eight anodes; U and V match only on 4-7. It may
  well be my offline index mapping rather than a defect — it is recorded so it is
  not rediscovered, and it is *not* claimed as a finding.
- **§7's aperture is measured on the doc 28 ambiguity population only**, never
  end-to-end through the tree.

**Round 4** (this one; §8): the owner's four questions, **no behaviour change** —
one doctest and three fields appended to round 2's env-gated dump line. It
answers all four and, in the course of answering (1), root-causes §4.1:

1. **§8.2 supersedes the plan.** Phase 1 is not making bad decisions below V; it
   is being fed a plane of zeros. `ImproveCluster_1::make_iblobs_improved`
   (`improvecluster_1.cxx:840`) indexes `IWirePlane::channels()` — the
   segment-0-only list of `AnodePlane.cxx:242-256` — by **wire index**, so on the
   16 PDVD planes carrying continuations the retile writes each slice's activity
   under the wrong channel and drops the top band outright. Measured in the
   retiled cluster: below V, **99.8 %** of points hold ≤1 non-zero plane, which
   `ncharge > 1` turns into charge 0 and no candidate at any threshold. The
   prediction was written before the run and holds on all three planes and both
   halves.
2. **It explains round 3's null result**, which §5.3 could only record: the fix
   and the defect live in different objects, and the knob structurally cannot
   reach the retile.
3. **The filters are cleared, on measurement not assumption** (§8.1): Phase 3's
   two clauses fire on 50.3 % and 6.8 % of terminals and their conjunction on
   **0** — by design. One latent sentinel-sign defect recorded, 0 fires here.
4. **§3's "SBND runs both off" was wrong** (stale by a month) and is corrected;
   the pre-prototype detector is uBooNE.
5. **Event 298595 is unmoved** (§8.3), which confirms doc 30's `fit_exclusion`
   attribution rather than competing with it — while the rest of that event moves
   a lot, including one newly STM-tagged cluster.

Still not done, and explicitly owed (carrying forward the round-3 list above):

- **The retile mapping fix itself** — §8.5 step 1. Not written this round: the
  owner asked for the analysis, explicitly not the code.
- **§6.1's density and localization metrics.** Only the gap metric exists. Every
  §6 lever admits more terminals, so shipping the redesign against coverage alone
  would grade it by a metric it cannot fail. The localization metric needs
  distance to the **fitted** skeleton, a PR product the phase dump does not carry.
- **Cluster 109's new STM tag on 039252/2** (§8.3) and **cluster 47's lost tag on
  039349/14** (§5.3) both need a hand scan before any default flip.
- **PDVD runs two terminal thresholds in one stage** (§8.4): 500 in
  `CreateSteinerGraph`, 4000 inside `ImproveCluster_2`, the second unreachable
  from jsonnet. Whether that is intended is an owner call; it is surfaced here,
  not silently resolved.
- Round 3's outstanding items are unchanged: no default flip anywhere, no PDHD
  event-level gate, the dead-blob sampler untouched, the anodes 0-3 `wire_index`
  question unexplained, §7's aperture never taken end-to-end through the tree.

**Round 5** (this one; §9): the retile mapping is fixed behind
`wrapped_channel_activity` (default OFF), and **the 108.5 cm gap closes to
1.8 cm** — 5 steiner points and 1 terminal below the vertex become 666 and 198.
Gates in §9.2, including byte-identical compiled configs for SBND and uBooNE
because this round edited the shared `cfg/pgrapher/common/clus.jsonnet`. The
composition finding is §9.3: the retile fix recovers **U** and round 3's
`wrapped_channel_charge` is what recovers **V**, so the two are complementary and
round 3 was necessary after all, merely blocked by a second defect downstream.

**Round 6** (this one; §10): the owner's hand scan of three Bee sets, and the
three code decisions it settled. Both downstream STM verdict changes are
adjudicated **in the fix's favour** (cl 47 is a TGM, cl 109 a plausible STM), so
both knobs are flipped to PDVD production; `ImproveCluster_2`'s terminal
threshold is made configurable and synced to `CreateSteinerGraph`'s;
`traj_degenerate_wcpts_fallback` is retired. The one negative verdict — the
terminals are too dense — is root-caused in §10.3: the local-max suppression is
scoped to a single blob, so terminal density *is* candidate-bearing blob density
(1.02 terminals per such blob, measured over 92 calls). C++ defaults are
unchanged everywhere; the flips are PDVD config only.

**Round 7**, in priority order:

1. **A cross-blob suppression pass** (§10.3, §10.7). The only lever that can
   thin the terminals without also destroying the coverage round 5 restored.
   Default-OFF knob, graded on all three §6.1 metrics at once. Per-blob is
   prototype-faithful, so this is a deliberate divergence under M15.
2. **Manifest-scale validation of the three round-6 changes.** They shipped ON
   for PDVD on two events and two hand-scanned clusters — enough to flip, not
   enough to leave unvalidated.
3. **Then reopen §6, or retire it.** Parked by the owner in round 6. The
   redesign's case on *this* event is gone — the criterion was never the problem
   here. What survives is doc 28's population evidence and §7's aperture result,
   both measured on the input point cloud and untouched by any of this. §6
   should be re-argued on that basis or retired; it should not be inherited.

Newly owed by round 6: the flips are PDVD-only and **PDHD still has no
event-level gate** for `wrapped_channel_charge` despite carrying 2.5× the orphan
fraction; the Q5 sync is measured on one event, where it moves exactly one
cluster; and `retile_cluster.cxx:433`'s twin remains unreachable, so its knob-ON
branch is still untested code.

Still owed, unchanged from round 5: the dead-blob sampler still carries the
round-3 defect; the anodes 0-3 `wire_index` question is unexplained; §7's
aperture was never taken end-to-end through the tree; and the base `RetileCluster`
path's activity carries a hit flag (1.0) rather than charge
(`retile_cluster.cxx:143`), so its charges were never meaningful anyway.
