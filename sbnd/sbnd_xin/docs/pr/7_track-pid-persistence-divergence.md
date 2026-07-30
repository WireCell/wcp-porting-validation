# 7 — track-PID persistence divergence: the proton that was identified and then discarded

Status: DIAGNOSED (root cause pinned with code + data evidence). Fix
proposed, NOT yet implemented — owner decision on §5. Supersedes doc pr/5's
"structural muon-vs-flat gate" reading of the evt 172230 electron/proton
mixup: the direction gate abstention is real, but the prototype has a
direction-free recovery that the toolkit port silently drops.

## 0. Repro block

```bash
TK=/nfs/data/1/xqian/toolkit-dev/toolkit
W=/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin/work-nuecc48-prsmoke2
# the discarding guard:
sed -n '1679,1694p' $TK/clus/src/PRSegmentFunctions.cxx
# prototype persists type unconditionally, gates only cal_4mom:
sed -n '1572,1581p;1626,1639p' $TK/prototype_base/pid/src/ProtoSegment.cxx
# our segment took the Track branch and exited with pdg=0:
grep "determine_direction: Track" $W/nupr_evt172230_dl_trace2/wct_trace.log | grep "len=9.75"
# its median dQ/dx (T_rec_charge, q = dQ*0.1-1000, dQ/dx = (q+1000)*10/nq):
#   -> 140695 e/cm = 3.27 x 43e3 = 2.51 x 56e3   (17 points, all >= 2.29 x 43e3)
```

## 1. Symptom

Evt 18253/1/172230, cluster 5030: a 9.7 cm track with a textbook Bragg peak
is labeled electron (pdg 11, "29 MeV"), and the traditional (geometric)
neutrino vertex lands on the Bragg-peak end. Doc pr/5 traced the toolkit
chain: `segment_do_track_pid` abstains (muon-vs-flat KS gate), pdg stays 0,
`examine_all_showers` sweeps the last non-shower pdg-0 segment to 11, zero
tracks -> all-showers vertexing -> PCA coin flip.

## 2. At what point would a track be identified as a proton? (census)

Prototype sites that assign `particle_type = 2212`:

| # | site | condition | needs direction? |
|---|------|-----------|------------------|
| 1 | `ProtoSegment.cxx:1224/1238` (do_track_pid) | proton wins KS score competition | YES — behind the muon-vs-flat direction gate |
| 2 | `ProtoSegment.cxx:1577` (determine_dir_track fallback) | pdg==0 and median dQ/dx > 1.75x MIP | **NO** |
| 3 | `ProtoSegment.cxx:1605/1612` (vertex-activity stubs <1.5cm) | median dQ/dx > 1.75x MIP | NO (dir from topology) |
| 4 | `NeutrinoID_track_shower.h:2076` (examine_direction, main vertex) | pdg==0, not flag_only_showers, median dQ/dx > 1.4x MIP | NO (at main vertex) |
| 5 | `NeutrinoID_track_shower.h:2101/2230` (in-shower / neighbor-proton reassign) | median dQ/dx > 1.3x MIP | NO |

Path 1 abstained for our proton (pr/5: smeared Bragg rise KS-matches flat
better than muon both ways — faithful prototype behavior). Path 2 is the
prototype's designed recovery for exactly this case — and our track passes
it overwhelmingly: median dQ/dx 140695 e/cm = **3.27x** the 43e3 threshold
scale (2.51x even against the SBND-calibrated 56e3), minimum point 2.29x,
vs the 1.75x cut. Path 4 was unreachable only because the cluster had
already gone all-showers — which itself is a consequence of path 2 failing.

## 3. Root cause

Toolkit `segment_determine_dir_track` (`PRSegmentFunctions.cxx`) computes
the fallback correctly (`:1628-1636` — faithful port, fires for our track)
but **persists the result only behind a direction test** (`:1679`):

```cpp
if (pdg_code != 0 && ((segment->dirsign() == 1 && end_n == 1) ||
                      (segment->dirsign() == -1 && start_n == 1))) {
    ... segment->particle_info(pinfo);  segment->particle_score(...);
}
```

The prototype persists type and mass **unconditionally** — `particle_type`
is a class member, assigned at `ProtoSegment.cxx:1577`, mass at
`:1626-1633` — and gates only the 4-momentum computation on the direction
test (`:1637-1639`):

```cpp
if ((flag_dir==1 && end_n == 1 || flag_dir==-1 && start_n ==1) && particle_type!=0){
    cal_4mom();
}
```

The port merged "compute 4-mom" and "store identity" into one conditional
(the member-variable -> ParticleInfo-object translation made storing an
explicit act, and it was placed inside the wrong if). Consequence: **every
segment identified by dQ/dx alone — i.e. without a confident KS direction —
loses its identity on function exit.** That is precisely the
Bragg-peaked-proton class: KS direction abstains (path 1), median-dQ/dx
catch fires (path 2), result discarded, pdg reads 0 downstream.

Evidence the discard happened for our segment: the `dl_trace2` trace shows
`determine_direction: Track nfits=17 nwcpts=23 len=9.75cm dirsign=0
dir_weak=0 start_n=2 end_n=1 pdg=0` — the Track branch ran (so the fallback
ran; it cannot return <1.75x with these 17 points in any trimming) yet the
segment exits with no particle info. Audit: this is the only conditionally
gated `particle_info` store in `PRSegmentFunctions.cxx` (the
shower-trajectory path at `:1758` stores unconditionally, matching the
prototype's forced 11).

Downstream cascade once the identity is dropped: pdg 0 -> swept to 11 by
`examine_all_showers` (prototype has the same sweep; but with pdg 2212 it
never fires) -> zero tracks -> `flag_save_only_showers` -> the main-vertex
proton catch (path 4) is explicitly bypassed in only-showers mode
(prototype `NeutrinoID_track_shower.h:2069-2085`: only-showers forces
pdg-0 segments to ELECTRON at the main vertex; the >1.4x-MIP proton branch
sits in the else) -> PCA vertexing, coin-flip end. Both pr/5 symptoms
(electron label AND wrong vertex) unwind from this one discard.

## 4. Why it hid

- The uBooNE qlport gate is toolkit-vs-toolkit byte identity; this
  divergence sits identically on both arms, so no gate ever failed.
- In tagger-compare-vs-prototype logs the effect surfaces as broad
  category diffs (ssm/numu particle-type columns), always confounded with
  upstream vertex/PID diffs — same story as the is_dir_weak divergence
  (doc pr/6 §2).
- pr/5's reading stopped at the KS abstention because the abstention IS
  faithful and numerically reproducible; the discarded recovery only shows
  when you ask "where would the prototype have caught it anyway" (the §2
  census) and check each site's port.
- The trace line prints `pdg=0` with no hint that a pdg was computed and
  dropped inside the call.

## 5. Proposed fix (systematic, knob-gated) — NOT yet implemented

Restore prototype persistence semantics in `segment_determine_dir_track`,
behind a config knob (C++ default false = current behavior, byte-identical):

```cpp
if (pdg_code != 0) {
    WireCell::D4Vector<double> four_momentum{};   // zero until direction is confident,
    if ((segment->dirsign() == 1 && end_n == 1) || // exactly like the prototype's
        (segment->dirsign() == -1 && start_n == 1)) // un-called cal_4mom()
        four_momentum = segment_cal_4mom(segment, pdg_code, particle_data, recomb_model, MIP_dQdx);
    segment->particle_info(std::make_shared<Aux::ParticleInfo>(pdg_code, mass, name, four_momentum));
    segment->particle_score(particle_score);
}
```

Threading identical to `dir_weak_use_score` (doc pr/6): knob on
TaggerCheckNeutrino -> PatternAlgorithms, key-suppression jsonnet, SBND
default ON, uBooNE default per fidelity study (expect improvement — this is
prototype parity). Validation: off-gate byte-identical (uBooNE 35-evt ZIPS
+ SBND 1000-evt hash set), knob-on rerun of evt 172230 (expect pdg 2212,
track survives, traditional vertex uses the Bragg constraint), 45
nu-candidate scan, uBooNE on-arm `fidelity_compare.py`.

Second, separable issue (real but NOT this event's cause): the MIP scale is
hard-coded `43e3` (uBooNE) at every PR-chain call site (~97 sites, pr/2
census) while SBND's calibrated value is 56000 e/cm (docs 47-48, already
configured for STM as `mip_dqdx=56000`). All the x1.2/1.3/1.4/1.5/1.75
median-dQ/dx thresholds are ~30% mis-scaled on SBND. Fix: thread a
`mip_dqdx` config through TaggerCheckNeutrino into the `MIP_dQdx` function
parameters that already exist (defaults preserved = byte-identical).
For this track it changes nothing (3.27x vs 2.51x, both >> 1.75) but it
recalibrates every marginal case.
