#!/usr/bin/env python3
"""doc qlmatch/35 F1(b) -- the light half of the flip-equivalence gate (d35/prereg.md sec 2 F1).

After the run_light_evt.sh default flip, over the 120 light events of pdvd/stm/events.txt:
  _tot     (flipped runner, no env, the _keep argument set via d32_light_arms.sh)  must equal  _q32ti  (the measured
           ToT light, made before the flip with PDVD_SAT_REPAIR_MODE=tot PDVD_HIT_INT_SAMPLES=1): opflash archive by
           abtest/hash_archive.py member content, and .wct-light.json after the tag in the output path is normalised;
           the config must say saturation_repair_mode "tot" and int_samples true;
  _q35esc  (flipped runner, PDVD_SAT_REPAIR_MODE=twoside)  must equal  _g31off  (the pre-flip knob-off light on this
           library, gate g31off == _keep) the same way, with neither key present.

    python3 d35_light_f1.py > ../d35/f1_light.txt
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PDVD = os.path.normpath(os.path.join(HERE, "..", "..", ".."))
HASH = os.path.normpath(os.path.join(PDVD, "..", "abtest", "hash_archive.py"))
PAIRS = [("_tot", "_q32ti", True), ("_q35esc", "_g31off", False)]


def light_events():
    for ln in open(os.path.join(PDVD, "stm", "events.txt")):
        f = ln.split()
        if len(f) >= 3 and not ln.startswith("#"):
            yield "%06d" % int(f[0]), f[2]


def arch_hash(p):
    out = subprocess.run([sys.executable, HASH, p], capture_output=True, text=True, check=True).stdout
    return out.split()[0]


def cfg(d, evt, tag):
    """the compiled config with this dir's own name in the output path normalised (a bare replace of '_tot' would
    also hit 'min_total_pe')."""
    return open(os.path.join(d, ".wct-light.json")).read().replace(f"_light{evt}{tag}/", "_light<EVT><TAG>/")


def main():
    W = os.path.join(PDVD, "work")
    fail = 0
    print("# doc qlmatch/35 F1(b) -- light after the runner-default flip; archive = hash_archive.py member content")
    for a, b, tot in PAIRS:
        n = same_arch = same_cfg = keys_ok = 0
        bad = []
        for r6, evt in light_events():
            da, db = f"{W}/{r6}_light{evt}{a}", f"{W}/{r6}_light{evt}{b}"
            n += 1
            try:
                ha = arch_hash(da + "/opflash_pdvd-wct.tar.gz")
                hb = arch_hash(db + "/opflash_pdvd-wct.tar.gz")
                ca, cb = cfg(da, evt, a), cfg(db, evt, b)
            except (OSError, subprocess.CalledProcessError) as exc:
                bad.append(f"{r6}_{evt}: {exc}")
                continue
            same_arch += ha == hb
            same_cfg += ca == cb
            has_tot = '"saturation_repair_mode" : "tot"' in ca
            has_int = '"int_samples" : true' in ca
            keys_ok += (has_tot and has_int) if tot else (not has_tot and not has_int)
            if ha != hb or ca != cb:
                bad.append(f"{r6}_{evt}: archive {'same' if ha == hb else 'DIFF'}, config {'same' if ca == cb else 'DIFF'}")
        want = "tot + int_samples present" if tot else "neither key present"
        ok = same_arch == n and same_cfg == n and keys_ok == n
        fail |= not ok
        print(f"{a} vs {b}: archives identical {same_arch}/{n}, configs identical {same_cfg}/{n}, "
              f"{want} {keys_ok}/{n} -> {'PASS' if ok else 'FAIL'}")
        for x in bad[:20]:
            print("   " + x)
    print(f"VERDICT F1(b): {'FAIL' if fail else 'PASS'}")


if __name__ == "__main__":
    main()
