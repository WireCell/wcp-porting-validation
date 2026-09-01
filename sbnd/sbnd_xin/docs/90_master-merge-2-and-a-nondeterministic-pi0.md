# doc 90 — second master merge (spng/OSP), and the bistable pi0 it exposed

**Status: merge CLEAN and committed (`50df4094`), NOT pushed pending the owner's
call on the gate below.** The uBooNE gate reports `rc=1` on 1 of 35 events.
That failure is **proven pre-existing and merge-independent**: the same commit
produces both outcomes. §3 is the evidence; §4 is the defect it exposed.

(Numbered 90, not 89: a concurrent session claimed 89 for
`89_cleanup-and-production-rebase.md` while this was being written. The arm
labels below keep their `doc89ub-` prefix — they are on-disk records and were
not renamed.)

## Repro

```bash
cd /nfs/data/1/xqian/toolkit-dev/toolkit && git log --oneline -1     # 50df4094
cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/qlport/scripts
./ab_check.sh doc89ub-mergereb doc89ub-basereb    # rc 1: zips 35/35, tagger 34/35
grep -m1 kine_pio_angle sweep/doc89ub-{basereb,mergereb}/22_6805/tagger_6805.log
```

## 1. What the merge contains

`37439d1c..e88a48aa`, 14 commits, +206/-24 over 15 files. Only **two** changed
files already existed in a form this tree runs:

| file | effect here |
|---|---|
| `sigproc/src/OmnibusSigProc.cxx` | adds a `wire_filter` array to the npz written when `dump_2d_spectra` is on. Flag is **default false** (`OmnibusSigProc.h:307`); no `cfg/` job sets it |
| `util/test/doctest_response.cxx` | upstream restored the `auto fr =` binding identically to our `5d924ab9`, so the merge auto-resolves |

Everything else is `spng/` (FrameToTdm, ResponseKernel, KernelConvolve, and the
spng cfg jobs): decon roll 128→129, Nyquist→sampling period, LMN resampling.
**`spng` is not built in this tree at all** (`local/lib/libWireCellSpng.so` does
not exist), so those are inert here.

**One live consumer, found only because doc 88's lesson was applied.**
`wcp-porting-img/pdvd/wct-nf-sp.jsonnet:91` sets `dump_2d_spectra: true`. That
PDVD diagnostic job's `sp_dump_*.npz` files gain one array. Debug artifact
only; no arithmetic changed. A `cfg/`-only grep would not have seen it —
exactly the failure mode of doc 87.

Merge is clean in both directions (`merge-tree` rc 0, identical tree
`edb76ba6`). Per-package suites all pass: util 42568, gen 223, sigproc 16491,
clus 2603, root 4035, aux 110738, iface 7. `prod_cfg_gate.py` **PASS 21/21**
vs `prod-2026-09-01c`.

## 2. The gate result

```
ab_check.sh doc89ub-mergereb doc89ub-basereb   -> rc 1
  ZIPS  : 35/35 content-identical
  TAGGER: identical=34 diff=1     (idx=22 ev=6805)
```

The Bee zips — the clustering output — are **always** identical, in every
comparison run for this doc. The single difference is four pi0 kinematic
variables in `track_com_5384_6805.root`:

| variable | state A | state B |
|---|---|---|
| `kine_pio_theta_2` | 5.76772e+01 | 9.13465e+01 |
| `kine_pio_phi_2` | 4.63050e+01 | 8.38583e+01 |
| `kine_pio_dis_2` | 1.48041e+01 | 5.18079e-01 |
| `kine_pio_angle` | 1.48058e+01 | 1.09512e+02 |

## 3. Why the merge is not the cause

The event is **bistable**, and the *same commit produces both states*:

| arm | commit | build | `kine_pio_angle` |
|---|---|---|---|
| `doc88ub-pre2` | 555d481f | pre-merge-1 lib snapshot | 14.81 |
| `doc88ub-box` | 555d481f | doc 88 build | 14.81 |
| `doc88ub-prac` | 555d481f | doc 88 build | 14.81 |
| `doc89ub-postmerge2` | 50df4094 | merge build | **109.51** |
| `doc89ub-basereb` | **555d481f** | fresh rebuild | **109.51** |
| `doc89ub-mergereb` | **50df4094** | fresh rebuild | 14.81 |

Rows 3 and 5 are the **same commit** and disagree. Rows 4 and 6 are the **same
commit** and disagree. No assignment of the difference to the merge survives
that table.

Two further controls were run before this was understood:

- **Reverting the OSP hunk alone** (`git show 37439d1c:sigproc/src/OmnibusSigProc.cxx`,
  rebuild `WireCellSigProc`) did **not** restore state A — so the one compiled
  change in the merge is not the trigger.
- **Rebuilding at the pre-merge commit** `555d481f` and re-running gave 109.51,
  i.e. the flip reproduces with the merge entirely absent.

A methodological note worth keeping: an earlier repeat test appeared to show
"three runs, two answers" and was **wrong** — it passed the filelist *index*
(22) where `wire-cell-uboone-tagger-compare` wanted the *subrun* (136), so the
comparison silently produced identical error logs. `tagger_log()` in
`ab_check.sh` reads the subrun from `meta.txt`; any hand-run comparison must do
the same. Six repeats with the correct prototype all gave 109.51.

## 4. The defect this exposed, and why no gate sees it

Event **5384-136-6805** has two pi0 solutions and selects between them in a way
that is not stable run-to-run. Clustering is deterministic (Bee zips identical
every time); the instability is downstream, in the tagger's pi0 kinematics.

`repeat_check.sh` — the determinism gate — **cannot detect this class of
defect**: it runs one event N times and counts *distinct Bee-zip content
hashes*. The Bee zip is identical in both states, so it reports "deterministic"
while `kine_pio_angle` moves by 95 degrees.

That is a real gap, not a nuisance:

- The pi0 campaign (docs pr/126–pr/141) selected its operating point on pi0
  kinematics. If this bistability affects SBND events too, some census rows
  are coin flips rather than measurements.
- Any uBooNE A/B gate has a per-run chance of a spurious 1-event FAIL, which
  trains the reader to wave through exactly the diff that matters.

**Not investigated further here** (CLAUDE.md §5.7: report a wrong-looking
physics number, do not tune it). The proximate suspects, in order: a
pointer-keyed container iterated in the pi0 pairing (M4 / the determinism
convention in §2 Code), or a genuine tie between two shower pairings broken by
thread arrival order — the sweep runs 6 events concurrently and wire-cell is
itself multi-threaded, while the single-run repeats were consistent.

## 5. Recommendation

1. **The merge is safe to keep.** Nothing this tree runs changed behaviour; the
   sole live effect is one extra debug array in a PDVD diagnostic npz.
2. **Do not "fix" the gate** by excluding event 6805. The correct next step is a
   determinism round on the pi0 kinematics: extend `repeat_check.sh` to hash the
   `T_kine`/pi0 branches of `track_com_*.root`, not just the Bee zip, then run
   N=8 on 6805 under both concurrent and serial execution to establish whether
   thread order is the discriminator.
3. **Check SBND for the same defect** before trusting pi0 census numbers to the
   last event — same instrument, same tagger code.
