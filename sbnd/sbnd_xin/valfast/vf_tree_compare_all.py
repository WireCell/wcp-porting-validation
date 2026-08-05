#!/usr/bin/env python3
"""valfast gate 1, WIDE edition -- every tree, exact, plus the calib dump.

Fork of vf_tree_compare.py (kept byte-for-byte: valfast/README.md's 2026-08-02
A/A' record cites its numbers, and the fork-by-duplication rule in CLAUDE.md
sec.2 forbids editing a file with live consumers).  Same CLI, same per-event
line shape, same 'mabc=/pctree=' tokens the valfast_compare*.sh drivers grep
for, so it is a drop-in under VF_CMP_WIDE=1.

FOUR differences, each closing a hole doc pr/37 sec.2 measured:

  1. ALL trees, not just T_tagger/T_kine.  tracking-pr.root carries seven --
     T_bad_ch Trun T_proj_data T_rec_charge T_proj T_tagger T_kine.  doc pr/31
     sec.12.3 gated on "T_tagger/T_kine leaf" and concluded its five fixes were
     null; the re-gate found shower_topo_reset (SBND ON) moving 23 of 501
     T_rec_charge/flag_shower entries on evt 52672 -- a third tree.  A gate that
     opens two of seven cannot see that class at all.

  2. EXACT compare, not sorted multisets.  vf_tree_compare.py sorts vector
     branches because a 2026-08-02 A/A' measured fill-order permutations
     everywhere.  Whether that is still true at 2457320d is what doc pr/37
     sec.2.5 measures -- see the a/b/c arms in pr37_a2_floor.sh.  Sorting hides
     a real reordering, and a reordered per-candidate feature vector IS a
     physics-visible change when a downstream reader indexes it.

  3. calib-pr-evt<ID>.json, split into tagger / kine / other keys.  This is the
     ONLY outlet for pr/36 F1's match_isFC (doc pr/36 sec.10.9: it is not
     booked in T_tagger), one of the 24 knobs ON at HEAD.  No valfast script
     mentions the file; arms must be built with PR_EXTRA_STAGES=pr_display for
     it to exist, and it is reported as an absent-side skip when they are not.

  4. Comparison goes through pr36_cmp.py's own _to_py/branch_equal.  doc pr/37
     sec.6.4: np.array_equal over jagged/object uproot branches manufactures
     phantom differences (~30 of them in one round), and repr() on an object
     array embeds a heap address (~235 phantom leaves in another).  Both
     survive an A-vs-A self-test.  Do not hand-roll a third one.

usage: vf_tree_compare_all.py <armA> <armB> <evt> [<evt> ...]
exit 0 iff every artifact compared is identical.
"""
import sys, os, json, subprocess

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pr36_cmp import branch_equal, calib_diff          # noqa: E402  (see note 4)

import uproot                                          # noqa: E402

HASH = "/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/abtest/hash_archive.py"


def archive_hash(path):
    if not os.path.exists(path):
        return "MISSING"
    out = subprocess.run([sys.executable, HASH, path], capture_output=True, text=True)
    return [line.split()[0:2] for line in out.stdout.strip().splitlines()]


def trees_diff(pa, pb):
    """[(tree, [branch,...]), ...] over EVERY tree; [] = identical.
    'MISSING'/'schema' entries are surfaced rather than skipped."""
    if not os.path.exists(pa) or not os.path.exists(pb):
        return [("file", ["MISSING"])]
    out = []
    with uproot.open(pa) as fa, uproot.open(pb) as fb:
        ta = {k.split(";")[0] for k in fa.keys(cycle=False)}
        tb = {k.split(";")[0] for k in fb.keys(cycle=False)}
        for tn in sorted(ta | tb):
            if tn not in ta or tn not in tb:
                out.append((tn, ["tree-presence"]))
                continue
            A, B = fa[tn], fb[tn]
            ka, kb = set(A.keys()), set(B.keys())
            moved = [f"+{k}" for k in sorted(ka ^ kb)]
            common = sorted(ka & kb)
            aa = A.arrays(common, library="np")
            bb = B.arrays(common, library="np")
            moved += [k for k in common if not branch_equal(aa[k], bb[k])]
            if moved:
                out.append((tn, moved))
    return out


def calib_state(da, db, evt):
    ca, cb = f"{da}/calib-pr-evt{evt}.json", f"{db}/calib-pr-evt{evt}.json"
    ea, eb = os.path.exists(ca), os.path.exists(cb)
    if not ea and not eb:
        return "absent", []
    if not (ea and eb):
        return "one-sided", ["absent " + ("A" if not ea else "B")]
    with open(ca, "rb") as f:
        a = f.read()
    with open(cb, "rb") as f:
        b = f.read()
    if a == b:
        return "identical", []
    tagger, kine, other = calib_diff(ca, cb)
    d = []
    if tagger:
        d.append(f"tagger{tagger[:6]}")
    if kine:
        d.append(f"kine{kine[:6]}")
    if other:
        d.append(f"OTHER{other[:6]}")
    return "DIFF", d


def main():
    armA, armB = sys.argv[1], sys.argv[2]
    evts = sys.argv[3:]
    n_mabc = n_pct = n_tree = n_cal = n_cal_seen = 0
    for evt in evts:
        da, db = os.path.join(armA, f"pr_evt{evt}"), os.path.join(armB, f"pr_evt{evt}")
        mab = archive_hash(f"{da}/mabc-pr.zip") == archive_hash(f"{db}/mabc-pr.zip")
        pct = (archive_hash(f"{da}/pctree-pr-evt{evt}.tar.gz")
               == archive_hash(f"{db}/pctree-pr-evt{evt}.tar.gz"))
        td = trees_diff(f"{da}/tracking-pr.root", f"{db}/tracking-pr.root")
        cs, cd = calib_state(da, db, evt)

        n_mabc += mab
        n_pct += pct
        n_tree += (not td)
        if cs != "absent":
            n_cal_seen += 1
            n_cal += (cs == "identical")

        ok = mab and pct and (not td) and cs in ("identical", "absent")
        if ok:
            print(f"OK  evt {evt}")
        else:
            tdesc = ";".join(f"{tn}{brs[:6]}" for tn, brs in td[:4])
            print(f"DIFF evt {evt}  mabc={'=' if mab else '≠'} "
                  f"pctree={'=' if pct else '≠'} trees={'=' if not td else '≠'} "
                  f"calib={cs}  {tdesc} {';'.join(cd)}")
    n = len(evts)
    print(f"SUMMARY {n} events | mabc identical {n_mabc}/{n} | pctree {n_pct}/{n} | "
          f"trees (ALL, EXACT) {n_tree}/{n} | calib {n_cal}/{n_cal_seen}"
          + (f" ({n - n_cal_seen} absent)" if n_cal_seen != n else ""))
    return 0 if (n_mabc == n and n_pct == n and n_tree == n and n_cal == n_cal_seen) else 1


if __name__ == "__main__":
    sys.exit(main())
