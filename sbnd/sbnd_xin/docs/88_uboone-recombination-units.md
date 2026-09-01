# doc 88 — the master merge silently changed 2/3 of uBooNE, and the unit convention that did it

**Status: CLOSED.** uBooNE restored **byte-identical** to its pre-merge output
(35/35 events). SBND production **unaffected and unchanged** (20 of 21 pinned
consumers byte-identical; the 21st is uBooNE's own job).
Toolkit: new component `PracticalBoxRecombination`.
Reference: `ref/prod-2026-09-01c/`.

Two changes go beyond the minimum uBooNE fix and are called out so they are not
discovered later: SBND's **dormant** `sbnd_box_recomb` is re-typed too (§5),
and the checked-in fixture `clus/test/data/uboone-mabc_config.json` plus two
`find_tn` names in clus doctests are re-typed with it — so those three tests
now exercise the Practical class rather than the mismatched combination.

## Repro

```bash
cd /nfs/data/1/xqian/toolkit-dev/wcp-porting-img/qlport/scripts

# the three arms (35 events each, ~1 min at concurrency 6)
./sweep_5384.sh doc88ub-prac 6                                     # fix
./sweep_5384.sh doc88ub-box  6                                     # post-merge as-is
LD_LIBRARY_PATH=/home/xqian/tmp/pr142-libsnap:$PWD/../../../local/lib \
  ./sweep_5384.sh doc88ub-pre2 6                                   # pre-merge binary

./ab_check.sh doc88ub-box  doc88ub-pre2   # rc 1 -- 23 of 35 events changed
./ab_check.sh doc88ub-prac doc88ub-pre2   # rc 0 -- 35/35 zips, 35/35 tagger logs

cd ../../sbnd/sbnd_xin && python3 scripts/cfg/prod_cfg_gate.py     # rc 0 vs prod-2026-09-01c
cd /nfs/data/1/xqian/toolkit-dev/toolkit && ./build/gen/wcdoctest-gen && ./build/clus/wcdoctest-clus
```

## 1. What happened

Doc 87 merged `origin/master` into `apply-pointcloud` (`f439a109`). One of its
commits, `e6fb7ef3` *"Fix 10x quenching-term unit error in Birks/Box
recombination"*, removed a `units::cm` (and a `/units::MeV`) from
`Gen::{Birks,Box}Recombination::operator()` and `::dE()`.

Doc 87 cleared that commit for **SBND**, correctly: SBND runs
`PowerBoxRecombination`, proven from the pinned compiled config. What doc 87
did **not** do was check the other caller. uBooNE does not use PowerBox — it
binds `BoxRecombination` outright:

```
qlport/uboone-mabc.jsonnet:866   "type": "BoxRecombination"
                                 A=1.0, B=0.255, Efield=0.273, rho=1.38, Wi=23.6e-6
```

Measured on the standard 35-event uBooNE manifest: **23 of 35 events (66%)
changed**. The Bee zips differ; the tagger-compare logs differ. This was live
for the whole of doc 87 and no gate covered it — `qlport/scripts/ab_check.sh`,
which CLAUDE.md §4 names as *the* uBooNE gate, was never run after the merge.

## 2. Why nothing caught it

Three separate guards were all structurally blind:

| guard | why it was silent |
|---|---|
| `prod_cfg_gate.py` 21/21 PASS (doc 87) | the change is in **C++**; the gate compares **compiled config**, which did not move |
| `doctest_pattern_recognition`, `doctest_tagger_check_neutrino` (both bind `BoxRecombination:box_recomb`) | they assert only `CHECK(std::isfinite(...))` — a 2x shift cannot fail them |
| `doctest_tracksegmentsampler` (upstream's own, asserts R in 0.6–0.8) | it uses the **C++ defaults**, which carry WCT units and are self-consistent |

Nothing in the tree asserted a recombination *value*. §6 fixes that.

## 3. Root cause — two unit conventions, not a bug on either side

Upstream is not wrong, and neither were our configs. They disagree about how
the parameters are expressed.

`Gen::BoxRecombination`'s own **C++ defaults carry WCT unit factors**:

```cpp
BoxRecombination(double Efield = 500 * units::volt / units::cm, double A = 0.930,
                 double B = 0.212 * units::gram/(units::MeV*units::cm2) * (units::kilovolt/units::cm),
                 double rho = 1.396 * units::gram / units::cm3, ...)
```

With those, upstream's post-`e6fb7ef3` formula is dimensionally correct and
gives R = 0.705 for a MIP. **Every detector config in this tree instead passes
bare numbers in practical units** — `Efield: 0.273`, `B: 0.255`, `rho: 1.38`
— and relied on the removed `*units::cm/units::MeV` to compensate. Since
`units::cm = 10` and `units::MeV = 1`, that factor is exactly **10**.

The practical convention is not a local quirk. It is the WCP prototype's:

```cpp
// prototype_base/wcp-uboone-bdt/inc/WCPLEEANA/cuts.h:394-395
float beta = 0.255;
float median_dedx = (exp((median_dqdx*43e3) * 23.6e-6*beta/1.38/0.273) - alpha)/(beta/1.38/0.273);
```

dQ/dx in e⁻/cm, dE/dx in MeV/cm — numerically identical to the **pre-merge**
WCT code. It is also LArSoft's (kModBoxA/kModBoxB), and it is what our own
`PowerBoxRecombination` uses internally (`RecombinationModels.cxx:161,166`
convert with `dE/units::MeV*units::cm/dX`). After the merge, `BoxRecombination`
and `PowerBoxRecombination` were on **inconsistent conventions**.

### The size of it, at uBooNE's parameters

Measured by calling the real `libWireCellGen.so` from each era:

| dQ/dx (e⁻/cm) | pre-merge dE/dx | post-merge dE/dx |
|---|---|---|
| 40,000 | 1.322 | 0.975 |
| **55,400 (MIP)** | **2.102** | **1.367** |
| 100,000 | 5.821 | 2.559 |
| 200,000 | 34.58 | 5.561 |

A 2.10 MeV/cm MIP at uBooNE's 0.273 kV/cm gives **55,362 e⁻/cm** pre-merge —
the familiar ~5×10⁴. Post-merge the same MIP reads 1.37 MeV/cm. The model
feeds `determine_direction`, `shower_determining_in_main_cluster`,
`determine_main_vertex`, `segment_cal_4mom` and `segment_cal_kine_dQdx`, so
this is direction, PID, vertex selection and kinematics — not a cosmetic
quantity.

## 4. Why a config-only fix was rejected — measured, not assumed

The obvious repair is to compensate in the config: scale `B` by 10, or write
the parameters with `wc.gram`/`wc.kilovolt` unit factors so the ratio comes out
right. Both give the **correct physics**. Neither is **bit-identical**.

Probing the real binaries at the project's `-O2`, 10 sample points
(6 forward, 4 inverse), comparing bit patterns against the pre-merge
`libWireCellGen.so`:

| candidate | points differing from pre-merge |
|---|---|
| config: `B` 0.255 → 2.55 | **4 of 10** (worst relative 4.4e-16) |
| config: unit-carrying `wc.gram/(wc.MeV*wc.cm2)*(wc.kilovolt/wc.cm)` | **5 of 10** |
| **new component `PracticalBoxRecombination`** | **0 of 10** |

4e-16 is physically irrelevant and would be a fine answer in most codebases.
It is not the bar here (CLAUDE.md §1): the gate is content-hash identity, and
pattern recognition has threshold comparisons that a last-ulp difference can
flip. So the fix is a component, not a number.

## 5. The fix

**New files only.** `gen/src/RecombinationModels.cxx` is the file upstream just
rewrote and will keep rewriting; editing it would conflict on every future
merge, and M10 says fork by duplication rather than modify a production file
with other live consumers.

- `gen/inc/WireCellGen/PracticalRecombinationModels.h`
- `gen/src/PracticalRecombinationModels.cxx`

`Gen::PracticalBoxRecombination` is the same Modified Box physics with
parameters in practical units. Its two method bodies are the pre-`e6fb7ef3`
arithmetic **transcribed verbatim** — same expression shapes, same operation
order — which is what makes it bit-identical rather than merely equal. The
source says so, because "simplifying" `units::MeV` (== 1) away would silently
reintroduce the ulp drift of §4.

Defaults are the canonical LArSoft ModBox point (A=0.93, B=0.212, E=0.5 kV/cm,
rho=1.396), i.e. R = 0.705 for a MIP.

Bindings changed (parameters untouched everywhere):

| file | change |
|---|---|
| `qlport/uboone-mabc.jsonnet:866` | `BoxRecombination` → `PracticalBoxRecombination` |
| `cfg/.../sbnd/clus.jsonnet` `sbnd_box_recomb` | same — the **dormant** `use_power_recomb=false` path |
| `clus/test/data/uboone-mabc_config.json` (×2) | the compiled fixture the three uBooNE doctests load |
| `clus/test/doctest_{pattern_recognition,tagger_check_neutrino}.cxx` | the `find_tn` names that must move with it |

The SBND `sbnd_box_recomb` node is only *selected* when `use_power_recomb` is
false, so it never reaches the compiled SBND config — confirmed by 20/21
pinned consumers being byte-identical. Re-typing it makes the knob's
documented promise honest again — but read it precisely. Doc 87 flagged two
comments claiming `use_power_recomb=false` "restores the byte-identical
pre-pr/10 config", which the merge had made false. After this change that arm
reproduces the pre-pr/10 **numbers**, because `PracticalBoxRecombination`
carries the pre-merge arithmetic. It is **no longer byte-identical in compiled
config**, because the pre-pr/10 config named `BoxRecombination` and this one
names `PracticalBoxRecombination`. The comments in `clus.jsonnet` and
`wct-pr-perevt.jsonnet` say exactly that; do not shorten either to "byte-identical".

## 6. Validation

| gate | result |
|---|---|
| `ab_check.sh doc88ub-prac doc88ub-pre2` | **PASS rc 0** — 35/35 Bee zips content-identical, 35/35 tagger-compare logs identical |
| `ab_check.sh doc88ub-box doc88ub-pre2` (control) | **FAIL rc 1** — 23/35 events differ. A gate that cannot fail proves nothing; this is the arm that shows the first one is not vacuous |
| bit probe vs pre-merge `libWireCellGen.so` | **0 of 10** sample points differ (§4) |
| compiled uBooNE config, before → after | **exactly 2 lines** — the type and the reference; every parameter unchanged |
| `prod_cfg_gate.py` | **PASS 21/21** vs `prod-2026-09-01c`; only `uboone.json` moved from `b` |
| `wcdoctest-gen` | 223/223 (16 cases) |
| `wcdoctest-clus` | 2603/2603 (235 cases) — the three uBooNE-fixture doctests now bind the Practical type |

**New test — `gen/test/doctest_practical_recombination.cxx`.** The gap §2
exposed was that no test asserted a recombination *value*. Three cases:
the LArSoft ModBox anchor (R = 0.7044 at 2.1 MeV/cm), prototype parity against
`cuts.h`'s expression at four dQ/dx points plus the pinned uBooNE MIP
(55362.11321505277 e⁻/cm), and a case asserting the two classes agree when each
is fed **its own** convention and disagree by ~2x when crossed. That last one
is the tripwire: if it ever starts passing, the classes have converged and this
doc's binding should be revisited.

**The library pin is proven, not assumed.** `doc88ub-box` and `doc88ub-pre2`
ran with the **same** (original) config and differ on 23/35 events; identical
config plus identical binary cannot do that, so the
`LD_LIBRARY_PATH=/home/xqian/tmp/pr142-libsnap` pin demonstrably took. The two
gates coming out opposite is the same argument from the other side: if the pin
had silently failed, `doc88ub-pre2` would equal `doc88ub-box` and Gate 1 would
have passed while Gate 2 failed.

**Bonus finding.** That snapshot is from Aug 30 and so predates every doc 87
commit (the MABC Bee guard, `save_in_scope`, the production flip). Gate 2
therefore shows those are **also inert on the uBooNE chain** — a stronger
result than was aimed for, and one doc 87 did not establish.

**Scope containment.** No file under `clus/src`, `clus/inc`, `img/`, `root/`,
`aux/`, `iface/`, `sigproc/` or `util/src` is touched — the toolkit diff is
two new `gen/` files, two `clus/test` fixtures, and comments. SBND's compiled
config is byte-identical. SBND arms were therefore **not** re-run; that is an
argument plus a compiled-config measurement, not a 308-event gate.

## 7. What this changes about doc 87

Doc 87's merge validation stands for SBND and is unaffected. Its gap was
scope: it gated 308 SBND events and never ran the uBooNE chain, then reported
the uBooNE exposure as conditional ("*if* anything ever ran it through
`NeedRecombModel` unnamed"). That was wrong in the direction that mattered —
the live uBooNE job names `BoxRecombination` explicitly, and two thirds of its
events had already moved.

The general lesson is narrower than "run more gates": **a merge that changes a
shared component must be gated on every detector that binds it, and binding
can be by explicit name in a config that lives in the other repo.** A grep
confined to `cfg/` cannot see `qlport/`.

## 8. Next

1. **Push** — both repos are committed and gated; the push is owner-gated
   (CLAUDE.md §5.6) and currently blocked on ssh agent availability.
2. **`BirksRecombination` has the same latent trap** and is unfixed. Nothing in
   either repo configures it today (grep: zero hits), so there is nothing to
   restore — but a future config written in practical units would hit exactly
   this. A `PracticalBirksRecombination` is ~30 lines if it is ever wanted;
   not written on speculation.
3. **Upstreaming.** `PracticalBoxRecombination` is additive and carries no new
   dependency. Whether to offer it upstream — or instead propose that
   `BoxRecombination` document its unit expectation — is an owner/team call
   (toolkit/CLAUDE.md: the human writes the PR preamble).
