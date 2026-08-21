# doc pr/108 — Is the exclusion fit's impact the same in the prototype and the toolkit? (2026-08-21)

**Status:** audit + two dedicated tests done; **no production change**. Test A (toolkit self-test)
PASSES exactly: the dQ/dx fit is bit-for-bit independent of the association pass (382 fits,
45 552 points, max|ΔdQ| = 0). Test B (prototype vs toolkit, exclusion MATCHED on uBooNE 5384,
four arms) shows the exclusion *mechanism* is the same but its *measured impact is not
functionally identical*: in the prototype, exclusion ON raises the charge within 3 cm of a
multi-prong junction on 9/11 junctions (+3…+15 %); in the toolkit the same switch gives +18…+33 %
on two events and 0…−4 % on three. Absolute ON-vs-ON junction charge agrees to 0.85–1.02 on 6/11
junctions, 0.72–0.75 on 6805 and 2.2× on 6806 (input-data treatment, §6). On SBND the toolkit's
exclusion-ON trajectory carries **13 % less charge within 1 cm of the target vertex** than OFF,
and pr/107's keep-all does not recover it (§5) — the opposite sign from what both implementations
do on uBooNE. Owner to direct the next step (§7).

Owner (2026-08-21): "double check between the prototype and toolkit implementations carefully …
1. organization of the input data for the fit? 2. possible sharing of dQ/dx fit? … if the fitted
track trajectory is the same, I assume the fitted dQ/dx would be the same with and without the
exclusion … Most of my existing work was done with this exclude fit off, until recently … the
main issue is really this exclude fit knob's impact. What we want is functionally identical."

## 0. Repro block

```bash
# toolkit 74d63484 + this round's TrackFitting.cxx (Test A hook, env-gated, no knob); lib 15:1x > src
cd toolkit && ./wcb build --notests -p && ./wcb install --notests -p
# Test A: sbnd_xin/scripts/pr108_testA.sh  (OFF gate 1 event vs work-pr107-off, then 3 events with the check)
WCT_DQDX_ASSOC_CHECK=1 SBND_FIT_EXCLUSION=true SBND_DQDX_FIT_KEEP_ALL_POINTS=true PR_JOBS=4 \
  ./run_pr_chain_batch.sh work-nuecc48-ql0819 work-pr108-assoccheck-nuecc48 data 10550 46363 81597
grep -h dqdx_assoc_check work-pr108-assoccheck-nuecc48/pr_evt*/wct_pr_evt*.log
# Test B, WCT arms (qlport/uboone-mabc.jsonnet + run_one.sh gained fit_exclusion / dqdx_fit_keep_all_points TLAs):
#   sbnd_xin/scripts/pr108_wct.sh -> qlport/scripts/sweep/pr108_wct_{off,on,onkeep}/<idx>_<ev>/track_com_5384_<ev>.root
#   (idx 1 4 6 16 22 23 = events 6505 6528 6532 6650 6805 6806; setarch -R; DL off)
# Test B, WCP arms (prototype_base/pid patched: WCP_FIT_EXCLUSION=0 forces exclusion OFF in do_multi_tracking;
#   rebuilt with ./waf-tools/waf build --targets=WCPPID,wire-cell-prod-nue-port in prototype-dev env):
#   sbnd_xin/scripts/run_wcp.sh on|off <evts> -> qlport/scripts/sweep/pr108_wcp_{on,off}/nue_5384_<sr>_<ev>.root
# comparators:
python3 sbnd_xin/scripts/pr108_fit_point_compare.py A.root B.root --kind-a wcp --kind-b wct --undo-u07   # per point
python3 sbnd_xin/scripts/pr108_junction_charge.py --ref WCP-on.root --arm L=file[:kind] ... --undo-u07    # charge near junctions
```

## 1. Audit: what is the same (file:line on both sides)

| item | toolkit | prototype | verdict |
|---|---|---|---|
| dQ/dx fit reads the association maps? | `TrackFitting.cxx:6245-7418` — zero uses of `m_3d_to_2d`/`m_2d_to_3d`/`associated_2d_points` | `PR3DCluster_multi_dQ_dx_fit.h` — zero uses of `map_3D_2D*_set` (not in the signature, `PR3DCluster.h:237`) | **same: never** |
| 2-D measurements | whole per-cluster charge map (`m_cluster_charge_data`, 6267-6320), every cell a row | whole `map_2D_*_charge` (`:141-173`), every cell a row | same |
| coupling window | `search_range=10` wires / 10·ticks (6784-6811) | `<= 10` channels & `<= 10` slices (`:371,390,411`) | same |
| sharing between segments | one simultaneous system; vertex = one shared column; 10 Gaussian sub-samples per point | same (`:96-117`, `:239-312`, `:804-808`) | same |
| regularisation | λ 0.0005·8/5 = 0.0008; close 0.15·5/3 = 0.25 / 0.45·5/3 = 0.75; dead 0.3/0.9; ×0.01 when `!flag_dQ_dx_fit_reg` | 0.0008; 0.25/0.75; 0.3/0.9; ×0.01 (`:757-797`) | same |
| uncertainties | rel 0.075/0.05, add 0/300 | same (`:53-57`) | same |
| end-point extension | `end_point_limit/2` = 0.3 cm (8521, never restored) | `end_point_limit` = 0.3 cm after round 2 (`:106`, `:188`) | same |
| shared-wire error inflation | `update_dQ_dx_data`, `share_charge_err=8000`, synced to the per-cluster map (6254-6263) | `update_data_dQ_dx_fit` 8000 (`_dQ_dx_fit.h:1067-1073`) | same rule, **different "other cluster" set (C3)** |
| exclusion `update_association` | interior points only, rounds 1-2, keep iff strictly nearer or < 0.3 cm; -1 sentinel → 1e9 (pr/98) | `multi_track_fitting.h:970-1096`, interior only, same rule | same |
| vertex associations | never exclusion-filtered, always stored (3648-3707) | same (`:901-963`) | same |
| pass between the last trajectory round and dQ/dx | third `form_map_graph(flag_exclusion)` (pr/28 T4) — drops zero-quantity points unless `dqdx_fit_keep_all_points` | none (`reset_fit_prop` = resize, `:175-188`) | same point set only with pr/107 ON |
| DL input cloud | vertices (fit point, dQ) then segment interior fits, `dQ·0.1−1000`, no filter | `NeutrinoID_DL.h:16-33`, identical | same |

So the owner's premise holds in both implementations: **the dQ/dx fit depends on the exclusion fit
only through the trajectory point set and positions** — exactly in the prototype, and in the toolkit
once pr/107 keeps every point. Test A proves the toolkit half numerically (§3).

Candidate divergences (measured or recorded, not acted on):
- **C1 — pr/98 §1's justification was wrong.** It states "the prototype's dQ/dx consumes the
  round-2 exclusion-filtered associations" — it consumes none. That sentence is why the third pass
  received `flag_exclusion`, and hence why the drop existed. Correction appended to pr/98 and to
  `do_multi_tracking_review.md` §4.3; `porting_dictionary.md` entry added.
- **C3 — "shared wire" set**: toolkit = a blob outside *all loaded* clusters (`update_dQ_dx_data`,
  `track_blobs_set` over `m_clusters`); prototype = an mcell outside *this* cluster. Measured on
  6805: channels with err = 8000 — WCP U/V/W 205/87/244 vs WCT 656/255/0 (§6).
- **C4 — prototype end-vertex regulariser bug**: `connected_vec` pushes `indices.size()-2` (a size,
  not an index) for the end vertex (`multi_dQ_dx_fit.h:723`); toolkit uses `fits[size-2].index`
  (7211). Recorded in `porting_dictionary.md` as an intentional divergence (the toolkit is right).
- **C5 — prototype U-plane /0.7 rescale** on uBooNE channel ranges (`:870-885`), not in the
  toolkit; undone in the comparator (`--undo-u07`) for Test B.
- **C6 — the uBooNE parity chain never matched exclusion**: `qlport/uboone-mabc.jsonnet` passed no
  `fit_exclusion` while the stored prototype references ran with it ON (28/30 sites). Every
  WCP-vs-WCT fit comparison before this doc compared exclusion-ON to exclusion-OFF. Fixed by the
  two new TLAs (default off, compiled JSON byte-identical — `cmp` against the pre-change tree).

## 2. Prototype exclusion switch (Test B infrastructure, owner-approved)

`prototype_base/pid/src/PR3DCluster_multi_track_fitting.h`, top of `do_multi_tracking`: if
`WCP_FIT_EXCLUSION=0` is in the environment, `flag_exclusion = false` (the two `break_segments`
sites already pass `false`). Unset ⇒ behaviour unchanged. Rebuilt `libWCPPID.so` +
`wire-cell-prod-nue-port` only (the full `waf build install` dies on an unrelated `paal` test compile
error). Run from the build dir with `LD_LIBRARY_PATH` = `prototype_base/build/*` + `install/lib64`.
**Check:** WCP-on re-run vs the stored references `prototype_base/nue_5384_*.root`: positions
identical on 6/6 events (282/282, 164/164, 257/257, 246/246, 208/208, 67/67 points at |Δ| = 0);
fitted dQ identical except 1–2 points per event on 3 events (max dq/q 1.65 / 4.81 / 6.58) — the
prototype's own run-to-run dQ/dx jitter (BiCGSTAB), worth knowing before reading any sub-% number.

## 3. Test A — toolkit self-test: dQ/dx is association-independent

`WCT_DQDX_ASSOC_CHECK=1` (debug-only env, no config, no knob; unset ⇒ no code path): after each
`dQ_dx_multi_fit`, snapshot every fitted dQ/dx, rebuild the associations with the **opposite**
`flag_exclusion` on the same point set (keep-all forced), refit, compare, restore.
nueCC48 10550 / 46363 / 81597 with `fit_exclusion` + keep-all ON: **382 fits, 45 552 segment
points, max|ΔdQ| = 0, max|Δdx| = 0, max|Δpos| = 0 on every call** (`work-pr108-assoccheck-nuecc48`).
OFF gate (env unset, 10550) vs `work-pr107-off`: PASS 2/2. The claim "same trajectory ⇒ same
dQ/dx with and without exclusion" is exact in the toolkit.

## 4. Test B — four arms on uBooNE 5384, per junction

Events with two ≥3-prong junctions in the main cluster (from the reference `T_rec_charge`): 6505,
6528, 6532, 6650, 6805, 6806. Arms: WCP-on (= stored reference), WCP-off (§2), WCT-on
(`fit_exclusion=true`), WCT-off (today's qlport state), WCT-on+keep (`dqdx_fit_keep_all_points`).
`T_rec_charge` q is the DL feature `dQ·0.1−1000` on both sides (`NeutrinoID.cxx:1883/1980`,
`UbooneMagnifyTrackingVisitor.cxx:387/476`); the comparators invert it to raw dQ.

**Trajectories.** Position-matched (nearest point, ≤0.5 cm): WCP-on vs WCT-on median |Δpos|
0.13–0.26 cm with signed medians ≈ 0 — the two curves coincide to within the 0.6 cm sampling
phase, on every event. Exclusion moves the interior of each prong by < 0.06 cm median on both
sides; near junctions median 0.13–0.31 cm on both sides.

**WCT-on vs WCT-on+keep**: identical on 5/6 events (0 points differ); 6528 adds 25 retained points
with every existing point unchanged. On uBooNE the pr/107 drop is almost absent (SBND: 443/47 events).

**Charge within 3 cm of each reference junction** (ΣdQ, raw, vertex row counted once; `pr108_junction_charge.py`, `--undo-u07`):

| evt J | WCP on | WCP off | Δ(off→on) | WCT on | WCT off | Δ(off→on) | WCT-on / WCP-on |
|---|---|---|---|---|---|---|---|
| 6505 J0 | 953 k | 871 k | +9 % | 968 k | 937 k | +3 % | 1.02 |
| 6505 J1 | 1118 k | 992 k | +13 % | 1057 k | 990 k | +7 % | 0.95 |
| 6528 J0 | 791 k | 750 k | +5 % | 673 k | 554 k | **+21 %** | 0.85 |
| 6528 J1 | 593 k | 604 k | −2 % | 576 k | 440 k | **+31 %** | 0.97 |
| 6532 J0 | 941 k | 798 k | **+18 %** | 847 k | 882 k | −4 % | 0.90 |
| 6532 J1 | 740 k | (vertex lost, 32 cm) | — | (no vertex within 3.8 cm, either arm) | — | — | — |
| 6650 J0 | 951 k | 920 k | +3 % | 825 k | 840 k | −2 % | 0.87 |
| 6650 J1 | 858 k | 795 k | +8 % | 802 k | 799 k | 0 | 0.93 |
| 6805 J0 | 491 k | 437 k | +12 % | 353 k | 365 k | −3 % | **0.72** |
| 6805 J1 | 433 k | 394 k | +10 % | 323 k | 334 k | −3 % | **0.75** |
| 6806 J0 | 703 k | 791 k | −11 % | 1547 k | 1039 k | **+49 %** | **2.20** |
| 6806 J1 | 604 k | 690 k | −12 % | 1406 k | 961 k | **+46 %** | **2.33** |

Within 1 cm the picture is the same with more scatter (table in `108_junction-charge.txt`).

Reading:
1. **Same sign on most junctions, different magnitude.** Prototype: ON > OFF on 9/11 (+3…+18 %).
   Toolkit: ON > OFF by +21…+49 % on 6528/6806, but −2…−4 % on 6532/6650/6805 where the prototype
   gains +3…+18 %. The exclusion's *impact* is therefore not functionally identical at the
   junction-charge level, even though the mechanism is line-for-line the same (§1).
2. **Where the toolkit's OFF arm loses badly, its fit is pathological**: 6528 WCT-off has 26
   negative fitted charges (segment 19009 at 1.6–7.3 cm from J0, dQ −1 000…−8 600) and the 6528
   WCP-off has none (census: WCP-on/off 0/0, WCT-on/off 14/26 negative points, 8 each at dQ = 0).
   So the toolkit's exclusion-OFF fit can fail at a junction where the prototype's does not — the
   configuration the owner's historical work ran in.
3. **Absolute ON parity** is 0.85–1.02 on 6/11 junctions, 0.72–0.75 on 6805 and 2.2× on 6806 —
   an input/fit-organisation difference, not an exclusion one (§6).

## 5. SBND — the same readout on the pr/106/107 arms (47 nueCC48 events, own pre-DL cloud)

Cloud charge within R of the target vertex, summed over the 47 events (raw dQ, ×10⁶):

| arm | R ≤ 1 cm | R ≤ 2 cm | R ≤ 3 cm |
|---|---|---|---|
| exclusion ON + drop (production) | 8.42 | 17.75 | 27.49 |
| exclusion ON + keep-all (pr/107) | 8.44 | 17.67 | 27.45 |
| exclusion OFF (global, pr/106 §9) | **9.70** | 18.97 | 28.55 |

On SBND the exclusion-ON *trajectory* carries 13 % less charge within 1 cm of the vertex than the OFF
one, and retaining the dropped points changes nothing (+0.2 %) — the pr/106 §9 "OFF gain" is a
trajectory effect of the exclusion in the toolkit on SBND, with the **opposite sign** from what both
implementations show on uBooNE (§4, ON > OFF). The prototype cannot run SBND, so the SBND-side
parity has to be argued from the uBooNE test, and on uBooNE the toolkit's exclusion delta already
does not track the prototype's event-for-event.

## 6. Input-data organisation (secondary, recorded)

`T_proj_data` totals (measured / predicted 2-D charge, channels with err = 8000):

| evt | WCP-on | WCT-on |
|---|---|---|
| 6505 | n 6666; U 11.95/10.69 M (96 shared) V 11.13/10.45 (0) W 15.00/10.92 (0) | n 6466; U 12.40/11.09 (371) V 11.12/10.49 (0) W 15.18/10.88 (574) |
| 6805 | n 6228; U 7.59/4.65 (205) V 7.41/5.81 (87) W 11.61/6.29 (244) | n 4949; U 8.56/5.61 (656) V 7.92/6.54 (255) W 10.87/6.60 (0) |
| 6806 | n 1759; U 2.54/1.99 (0) V 7.06/2.04 (0) W 0.87/0.83 (0) | n 1186; U 2.53/2.71 (0) V 6.66/2.70 (18) W 0.84/0.81 (0) |

The measured maps differ by 5–13 % in extent and content (different channel sets, C3's shared-wire
set differs by hundreds of channels), so absolute dQ parity of 10–30 % (6805) is input-side; 6806's
2.2× sits on a V-plane overlap region (7.06 M measured vs 2.0 M predicted on both sides) where the
fits diverge. 6528 WCT `charge_pred` is ~0 on all planes — the toolkit's `T_proj_data` prediction
dump is broken for that event (dump only; the fit itself is fine). Not pursued this round.

## 7. Open / owner decision

- The exclusion mechanism is the same; its measured impact is not. The leads, in order: (a) the
  toolkit's exclusion-OFF fit producing negative charges at a junction where the prototype's does not
  (6528) — i.e. the OFF path, the one most SBND work was validated in, is the more suspicious side;
  (b) the SBND sign flip (§5); (c) C3 (shared-wire set) for absolute parity.
- A larger uBooNE sample (all 17 events with a junction) would turn §4 into statistics; the 6-event
  set is indicative.
- Nothing flipped; `dqdx_fit_keep_all_points` stays OFF; the Test A hook is debug-only.

Sidecars: `108_junction-charge.txt` (four arms, R = 1/2/3), `108_point-deltas.txt`, scripts
`pr108_fit_point_compare.py`, `pr108_junction_charge.py`, `pr108_testA.sh`, `pr108_wct.sh`,
`run_wcp.sh`; arms `qlport/scripts/sweep/pr108_{wct_off,wct_on,wct_onkeep,wcp_on,wcp_off}`,
`sbnd_xin/work-pr108-{off1,assoccheck}-nuecc48`.

## 8. Stage-by-stage dump (owner 2026-08-21: "track down where the ON-vs-OFF delta diverges")

Both sides now carry a debug-only, env-gated dump of every trajectory round (`WCT_TRAJ_DUMP` /
`WCP_TRAJ_DUMP`, same record layout; toolkit `TrackFitting::traj_dump_fits` + a per-point record in
`form_map_graph`; prototype `wcp_traj_dump_fits` + a record in `form_map_multi_segments`): per
interior point the association cell counts before exclusion, after `update_association`, after
`examine_point_association`, the live-plane quantity, kept/dropped, `dis_cut`; and the fitted
positions (+dQ, dx) after round 1, round 2 and the dQ/dx fit.  Arms re-run with the dump
(`qlport/scripts/sweep/pr108e_{wct_off,wct_on,wcp_on,wcp_off}`; `scripts/pr108_stage_diff.py`; full
output `108_stage-diff-all.txt`, summary `108_stage-summary.txt`).  The call compared on each side is
the last `do_multi_tracking` whose final trajectory reaches the junction.

Per junction, first association round (points within 3 cm; cells stripped by exclusion / cells
associated, WCP-on vs WCT-on), then the final near-junction ΣdQ delta off→on on each side:

| junction | map1 stripped / assoc (WCP / WCT) | map2 stripped / assoc | WCP off→on | WCT off→on | fit3 \|Δpos\| WCP-vs-WCT (ON / OFF) |
|---|---|---|---|---|---|
| 6505 J0 | 260/1258 vs 204/1283 (n 6/6) | 115/1114 vs 154/993 | +13 % | +7 % | 0.18 / 0.08 |
| 6505 J1 | 474/1480 vs 509/1603 (7/7) | 286/1439 vs 423/1519 | +15 % | +12 % | 0.26 / 0.30 |
| 6528 J0 | 386/1249 vs 464/1057 (7/6) | 229/1218 vs 299/1060 | +1 % | **+27 %** | 0.30 / 0.30 |
| 6528 J1 | 374/1121 vs 351/753 (6/4) | 216/902 vs 279/782 | −4 % | **+34 %** | 0.29 / 0.31 |
| 6532 J0 | 261/1127 vs 145/1017 (5/5) | 139/1018 vs 226/1025 | **+29 %** | −3 % | 0.18 / 0.17 |
| 6650 J0 | 257/879 vs 468/1243 (6/8) | 158/968 vs 419/1303 | +4 % | −8 % | 0.26 / 0.31 |
| 6650 J1 | 315/1058 vs 244/1087 (7/6) | 127/946 vs 239/1009 | +2 % | 0 | 0.24 / 0.11 |
| 6805 J0 | 114/840 vs 48/818 (4/3) | 62/837 vs 48/560 | +12 % | −2 % | 0.19 / 0.11 |
| 6805 J1 | 114/840 vs 48/552 (4/2) | 62/774 vs 48/565 | +15 % | −2 % | 0.19 / 0.13 |
| 6806 J0 | 456/1413 vs 447/1236 (7/6) | 211/1042 vs 281/1058 | −13 % | **+74 %** | 0.19 / 0.31 |
| 6806 J1 | 423/1094 vs 409/1048 (5/5) | 189/809 vs 258/829 | −14 % | **+73 %** | 0.19 / 0.37 |

Where the divergence is and is not:
- **Association stage — same.** With exclusion ON the two implementations associate the same number
  of cells per point (within the 0.2 cm sampling phase), strip comparable numbers of cells
  (6806: 456 vs 447; 6528 J1: 374 vs 351; 6505 J1: 474 vs 509), keep the same points, and compute the
  same live-plane quantities (`<q>` within 0.05).  The toolkit strips more on the second round at most
  junctions (its round-2 trajectory is re-sampled more finely: n 14–20 vs 13–17 points) — a
  difference in `organize_segments_path_2nd` sampling, not in the exclusion rule.
- **Trajectory positions — same to the sampling phase.** Final fitted positions agree at
  0.08–0.37 cm median on every junction, ON and OFF alike; the ON→OFF displacement within each side
  is also ≤ 0.3 cm.
- **dQ/dx solution — where it diverges.** With trajectories this close, 6806's +74 % (WCT) vs −13 %
  (WCP) and 6528's +30 % vs 0 come out of the charge *solution*, not the trajectory.  6806's
  junction sits where the W plane is dead (`T_proj_data` W: 0 cells on WCT, 162 zero-charge cells on
  WCP) and the V plane is heavily overlapped (measured 3.8 M vs predicted 1.3–1.8 M within ±6
  channels): a two-plane, overlapped system is ill-conditioned, and there WCT-ON over-predicts U
  (1458 k predicted vs 1195 k measured) while WCT-OFF (993 k) and WCP-ON (767 k) do not — every point
  within 2 cm of J0 carries ~2× the charge in WCT-ON (seg 1 pt 1: 115 k vs 60 k OFF vs 52 k WCP).
  The PR topology also differs: the prototype's final fit of that cluster has 11 segments / 64 points,
  the toolkit's 4 / 50.
- On SBND this is the same class of place (junction, shared cells, one plane weak) where the
  ON trajectory loses 13 % of the charge within 1 cm (§5).

So the exclusion *mechanism* is parity-exact through the association stage; what is not identical is
how the dQ/dx solution (and the pattern recognition that decides how many prongs share the junction)
responds to the exclusion-ON trajectory at ill-conditioned junctions.  That is the place to look next:
the regulariser/`connected_vec`/dead-plane weights path of `dQ_dx_multi_fit` at a junction with a dead
or overlapped plane (C4 is exactly there), and why the toolkit's PR keeps fewer segments at 6806.

Answer to the owner's question "what did pr/107 change then?": it changed the *point set* given
to the fit (443 junction points un-deleted), not the fit given a point set; the retained points took
the neighbouring prong's overlap charge and moved the prongs' early dQ/dx, which the EM-shower
clustering reads.  Consistent with Test A.
