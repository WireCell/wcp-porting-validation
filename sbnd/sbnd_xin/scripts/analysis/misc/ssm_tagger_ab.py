#!/usr/bin/env python3
"""Compare T_tagger between two SSM knob arms (same binary).

arm A: no ssm_target_dir key (uBooNE default)
arm B: ssm_target_dir=[0,0,1]  ->  every ssm_*_angle_target must equal the
       corresponding ssm_*_angle_z bit-for-bit, and nothing else may move.
"""
import uproot, numpy as np, sys

A, B = sys.argv[1], sys.argv[2]
ta = uproot.open(A)["T_tagger"]; tb = uproot.open(B)["T_tagger"]
ka, kb = sorted(ta.keys()), sorted(tb.keys())
assert ka == kb, "branch sets differ"
da = ta.arrays(ka, library="np"); db = tb.arrays(kb, library="np")

moved, same = [], 0
for k in ka:
    va, vb = np.asarray(da[k]), np.asarray(db[k])
    if va.dtype == object:  # jagged branch: compare entry by entry
        eq = len(va) == len(vb) and all(np.array_equal(x, y) for x, y in zip(va, vb))
    else:
        eq = va.shape == vb.shape and np.array_equal(va, vb)
    if eq:
        same += 1
    else:
        moved.append(k)
# doc pr/94 Phase 5: the [0] below is only a SAMPLE value printed next to a
# full-array comparison, so it stays correct when T_tagger/T_kine go multi-row
# (one entry per in-beam-window bundle under nu_per_bundle) -- the identity
# test itself already compares the whole array.  Reporting scripts that return
# one number per event use scripts/pr94_rows.primary_index() instead.
print(f"branches: {len(ka)}   identical: {same}   moved: {len(moved)}")
for k in moved:
    print(f"  MOVED {k:34s} A={np.asarray(da[k])[0]!r}  B={np.asarray(db[k])[0]!r}")

print("\nidentity check (B: target must == z):")
ok = True
for pre in ["ssm_angle_to", "ssm_nu_angle", "ssm_con_nu_angle", "ssm_prim_nu_angle", "ssm_track_angle"]:
    kt, kz = pre + "_target", pre + "_z"
    if kt not in ka:
        continue
    t, z = np.asarray(db[kt])[0], np.asarray(db[kz])[0]
    hit = (t == z)
    ok &= bool(hit)
    print(f"  {kt:34s} {t!r} vs {kz} {z!r}  {'EQUAL' if hit else 'DIFFER'}")
print("\nidentity:", "PASS" if ok else "FAIL")

print("\narm A target/absorber features (uBooNE default vectors):")
for k in ka:
    if "target" in k or "absorber" in k:
        print(f"  {k:34s} {np.asarray(da[k])[0]!r}")
