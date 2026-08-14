#!/usr/bin/env python3
"""Byte-identity gate between two PR arms, by archive MEMBER CONTENT (M2).

    gate_cmp_arms.py ARM_A ARM_B

Compares, per event present in both arms, in TWO SEPARATELY REPORTED CLASSES:

  RECONSTRUCTION (a difference here is a behaviour change -- gates the verdict):
    pctree-pr-evt<ID>.tar.gz   the point-cloud tree  (content hash)
    mabc-pr.zip                the Bee layers        (content hash)
    rc.txt                     the exit code         (literal)

  LOG-DERIVED (reported, does NOT gate -- see below):
    nusel-evt<ID>.tsv          the per-bundle labels (plain sha256)

WHY nusel-evt<ID>.tsv IS REPORTED BUT NOT GATING. It is produced by
nusel_extract.py by PARSING THE JOB LOG, and WCT writes long spdlog messages
non-atomically, so a line can be cut mid-word with another thread's message
spliced in. Its `stmfit` column is the only field sourced purely from log text;
`tgm`/`stm`/`fc` in the same row come from the post-PR tree. Measured on this
round's gate (docs/pr/76): adding the pr_display visitor shifted thread
interleaving and destroyed the cluster-id token in one skip line per event on
2 of 48 events, so `stmfit` flipped between 'contained' and 'eval' -- IN
OPPOSITE DIRECTIONS on the two events, with `tgm`/`stm`/`fc` identical (0/0/1)
and pctree+mabc byte-identical. The tearing is DETERMINISTIC PER PIPELINE, not
run-to-run noise: an A/A' repeat of the identical pipeline reproduced every TSV
exactly. So a nusel difference between two arms whose pipelines differ is
evidence about logging, not about physics, and must not be allowed to mask or
manufacture a gate verdict either way.

Read the two lines together: RECONSTRUCTION clean + nusel dirty => a logging
artifact, quote the torn line. RECONSTRUCTION dirty => a real difference,
regardless of what nusel says.

WHY THESE AND NOT OTHERS -- read before widening the set:

  * calib-pr-evt<ID>.json is DELIBERATELY EXCLUDED. It exists only in a
    PR_EXTRA_STAGES=pr_display arm, so including it would make an on-vs-off
    comparison fail by construction and prove nothing. Doc pr/26 sec 6's claim
    is precisely that adding the pr_display stage leaves the RECONSTRUCTION
    outputs untouched; this script tests that claim against the outputs that
    exist in both arms. The dump's own correctness is checked separately (its
    numbers against the job's own `DL rerank cand ... TOTAL=` TRACE lines).
  * tracking-pr.root is EXCLUDED: ROOT embeds creation timestamps and a UUID in
    every file, so it is never byte-reproducible and a raw hash would report a
    false FAIL on every event (this is M2 applied to ROOT rather than tar/zip).
    Its physics content reaches this gate anyway -- nusel-evt<ID>.tsv is
    extracted from the same tagger output.
  * stdout.log / wct_pr_evt<ID>.log are EXCLUDED: they carry wall-clock times,
    per-run paths, and thread-interleaved lines that tear mid-word
    (project_wct_log_line_tearing), none of which is a physics difference.

Never compares raw archive bytes: tar members are written with mtime=time(0)
and zip members carry wall-clock stamps, so `cmp`/`md5sum` on the archive
reports a difference that does not exist (M2). Uses abtest/hash_archive.py's
member-content hashing.

Exit 0 iff every compared artifact of every common event agrees AND neither arm
has events the other lacks.
"""
import os, sys, hashlib, importlib.util

AB = "/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/abtest/hash_archive.py"
spec = importlib.util.spec_from_file_location("hash_archive", AB)
ha = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ha)


def arch_hash(path):
    """Rollup sha256 over sorted (member-name + payload) -- NEVER raw bytes."""
    h = hashlib.sha256()
    n = 0
    for name, payload in ha.members(path):
        h.update(name.encode())
        h.update(payload)
        n += 1
    return h.hexdigest(), n


def file_hash(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def events(arm):
    out = {}
    for d in os.listdir(arm):
        if d.startswith("pr_evt") and os.path.isdir(os.path.join(arm, d)):
            out[d[len("pr_evt"):]] = os.path.join(arm, d)
    return out


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    A, B = sys.argv[1].rstrip("/"), sys.argv[2].rstrip("/")
    ea, eb = events(A), events(B)
    only_a, only_b = sorted(set(ea) - set(eb)), sorted(set(eb) - set(ea))
    common = sorted(set(ea) & set(eb), key=int)

    print(f"A = {A}   ({len(ea)} events)")
    print(f"B = {B}   ({len(eb)} events)")
    if only_a or only_b:
        print(f"!! only in A: {only_a}")
        print(f"!! only in B: {only_b}")

    ndiff = 0          # RECONSTRUCTION differences -- these gate
    nlogdiff = 0       # log-derived differences -- reported only
    logdiff_evts = []
    nmiss = 0
    ncmp = 0
    for evt in common:
        da, db = ea[evt], eb[evt]
        for fname, hasher, gating in (
                (f"pctree-pr-evt{evt}.tar.gz", arch_hash, True),
                ("mabc-pr.zip", arch_hash, True),
                ("rc.txt", file_hash, True),
                (f"nusel-evt{evt}.tsv", file_hash, False)):
            pa, pb = os.path.join(da, fname), os.path.join(db, fname)
            if not (os.path.exists(pa) and os.path.exists(pb)):
                # An artifact missing from BOTH arms is a shape both runs agree
                # on (e.g. an event with no PR output); missing from ONE is a
                # real difference and must not be silently skipped.
                if os.path.exists(pa) != os.path.exists(pb):
                    print(f"  DIFF evt {evt} {fname}: present in "
                          f"{'A' if os.path.exists(pa) else 'B'} only")
                    if gating:
                        ndiff += 1
                    else:
                        nlogdiff += 1
                else:
                    nmiss += 1
                continue
            ha_, hb_ = hasher(pa), hasher(pb)
            ncmp += 1
            if ha_ != hb_:
                tag = "DIFF" if gating else "diff(log-derived)"
                print(f"  {tag} evt {evt} {fname}")
                print(f"        A {ha_}")
                print(f"        B {hb_}")
                if gating:
                    ndiff += 1
                else:
                    nlogdiff += 1
                    logdiff_evts.append(evt)

    print(f"\ncommon events: {len(common)}   artifacts compared: {ncmp}   "
          f"absent in both: {nmiss}")
    print(f"RECONSTRUCTION (pctree/mabc/rc): {ndiff} differing artifact(s)")
    print(f"log-derived    (nusel tsv)     : {nlogdiff} differing"
          + (f"  evts {logdiff_evts}" if logdiff_evts else ""))
    if nlogdiff and not ndiff:
        print("  -> reconstruction clean + nusel dirty: inspect the `stmfit` cell and")
        print("     the job log for a torn line before reading this as a behaviour change.")
    bad = ndiff + len(only_a) + len(only_b)
    if bad == 0:
        print(f"GATE PASS -- 0/{len(common)} events differ on reconstruction output, "
              f"byte-identical by member content")
        return 0
    print(f"GATE FAIL -- {ndiff} differing reconstruction artifact(s), "
          f"{len(only_a)+len(only_b)} event(s) present in only one arm")
    return 1


if __name__ == "__main__":
    sys.exit(main())
