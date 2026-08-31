#!/usr/bin/env python3
"""Prove every .groups/g<N>.tar.gz in the 2026-08-31 removal set is a pure
duplicate of a KEEP Q/L root, so the archive step may DROP it instead of
tarring 5 GiB of copied pctree data into the record layer.

WHY THIS SCRIPT EXISTS.  archive_records_*.py splits every file in a retiring
arm into HEAVY (a reproducible data product -> DROPPED) and everything else
(-> ARCHIVED verbatim into the record tar).  HEAVY is a list of filename
patterns, and a pattern it does not know defaults to "record".  The 08-25 round
justified reusing HEAVY unchanged with a census: "ZERO unclassified file above
5 MiB".  That census was true of ITS removal set and is false of this one --
none of its arms was group-mode, and doc 84 round 3's census arms are.  The
first plan run put 16.06 GiB into the record layer, of which 4.96 GiB was 188
.groups/g<N>.tar.gz files, the group INPUT archives: bundles of Q/L pctrees
handed to a group-mode PR run.

That is duplicated data, not a record -- but "is a duplicate" is exactly the
kind of claim that must be measured rather than argued, because the copy that
would survive lives in a DIFFERENT arm and the round is what makes the
duplication load-bearing.  Same shape as doc 81 sec 7's 24536/24536 proof for
the imaging hubs: identity of member content, member by member.

METHOD.  For each g<N>.tar.gz: read its sibling g<N>-rse.json for the event
list the group carries, hash every member, then hash the members of the
corresponding per-event pctree-evt<E>.tar.gz under the KEEP Q/L root, and
require every group member to be present with an identical SHA-256.  A group
tar with even one unmatched member is reported and the round must keep it.

The KEEP root for an arm is derived from its own sample suffix, and the result
is REFUSED if that root is not in this round's KEEP set -- a duplicate of
something also being deleted is not a duplicate worth anything.

Writes state-20260831/group-dupes.tsv (one row per group tar) and exits
non-zero if any tar is not fully accounted for.  Reads only.
"""
import os, sys, json, glob, tarfile, hashlib, re
from concurrent.futures import ProcessPoolExecutor

ROOT = "/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin"
STATE = os.environ.get("RETIRE_STATE",
                       os.path.join(ROOT, "scripts", "retire", "state-20260831"))
JOBS = int(os.environ.get("RETIRE_JOBS", "12"))
os.chdir(ROOT)

plan = json.load(open(os.path.join(STATE, "plan.json")))
R, KEEP = set(plan["R"]), set(plan["KEEP"])

# arm sample suffix -> the Q/L root that survives this round
QL_ROOT = {
    'mcp1k':   'work-mcp1k-grp0825',
    'mcp2k':   'work-mcp2k-grp0825',
    'ncpi0':   'work-ncpi0-grp0825',
    'nuecc48': 'work-nuecc48-grp0825',
}


def sample_of(arm):
    for s in QL_ROOT:
        if arm.endswith('-' + s):
            return s
    return None


def members(path):
    out = {}
    with tarfile.open(path) as t:
        for ti in t:
            if ti.isfile():
                out[os.path.basename(ti.name)] = hashlib.sha256(
                    t.extractfile(ti).read()).hexdigest()
    return out


def check(job):
    arm, tgz = job
    s = sample_of(arm)
    src = QL_ROOT.get(s)
    rse = tgz.replace('.tar.gz', '-rse.json')
    try:
        got = members(tgz)
    except Exception as e:
        return (arm, os.path.basename(tgz), src or '-', 0, 0, 0, f"unreadable: {e}")
    if src is None:
        return (arm, os.path.basename(tgz), '-', len(got), 0, 0, "no sample suffix -> no KEEP root")
    if not os.path.exists(rse):
        return (arm, os.path.basename(tgz), src, len(got), 0, 0, "no g<N>-rse.json event list")
    evts = list(json.load(open(rse)))
    want = {}
    for e in evts:
        f = os.path.join(src, f'ql_evt{e}', f'pctree-evt{e}.tar.gz')
        if os.path.exists(f):
            try:
                want.update(members(f))
            except Exception as ex:
                return (arm, os.path.basename(tgz), src, len(got), len(evts), 0,
                        f"source unreadable: {ex}")
    match = sum(1 for k, v in got.items() if want.get(k) == v)
    note = "OK" if match == len(got) and got else f"{len(got)-match} member(s) unmatched"
    return (arm, os.path.basename(tgz), src, len(got), len(evts), match, note)


if __name__ == "__main__":
    jobs = []
    for arm in sorted(R):
        for tgz in sorted(glob.glob(os.path.join(arm, '.groups', 'g*.tar.gz'))):
            jobs.append((arm, tgz))
    if not jobs:
        # 2026-08-31 fork: the 08-29 original returned here WITHOUT writing the
        # proof, so retire_*.sh interlock 5 ("group-dupes.tsv MISSING") refused
        # a round whose group class is legitimately empty -- and the only ways
        # out were to hand-write an evidence file or to edit the interlock,
        # both of which defeat it.  An empty class is a RESULT and gets
        # recorded like any other: the file is written by the checking code,
        # header-only, and says so.
        os.makedirs(STATE, exist_ok=True)
        with open(os.path.join(STATE, "group-dupes.tsv"), "w") as fh:
            fh.write("# verify_group_dupes_20260831.py: 0 .groups/g*.tar.gz in the\n"
                     "# removal set (%d arms scanned) -- the group-input class is EMPTY\n"
                     "# this round, so archive_records drops 0.00 GiB of it and there is\n"
                     "# no duplication claim to measure.\n"
                     "arm\ttar\tmembers\tmatched\tverdict\n" % len(R))
        print("no .groups/g*.tar.gz in the removal set -- empty class recorded in "
              "group-dupes.tsv (%d arms scanned)" % len(R))
        sys.exit(0)

    arms = sorted({a for a, _ in jobs})
    print(f"{len(jobs)} group input archives across {len(arms)} arm(s), {JOBS}-way")
    bad_root = [a for a in arms if QL_ROOT.get(sample_of(a)) not in KEEP]
    if bad_root:
        print(f"!! these arms' Q/L roots are not in KEEP: {bad_root}")
        sys.exit(1)

    with ProcessPoolExecutor(max_workers=JOBS) as ex:
        rows = list(ex.map(check, jobs))

    os.makedirs(STATE, exist_ok=True)
    out = os.path.join(STATE, "group-dupes.tsv")
    with open(out, "w") as fh:
        fh.write("arm\tgroup_tar\tkeep_ql_root\tmembers\tevents\tmatched\tnote\n")
        for r in sorted(rows):
            fh.write("\t".join(str(x) for x in r) + "\n")

    tot_m = sum(r[3] for r in rows)
    tot_k = sum(r[5] for r in rows)
    bad = [r for r in rows if r[6] != "OK"]
    by_arm = {}
    for r in rows:
        a = by_arm.setdefault(r[0], [0, 0, 0])
        a[0] += 1; a[1] += r[3]; a[2] += r[5]
    for a in sorted(by_arm):
        n, m, k = by_arm[a]
        print(f"  {a:28s} {n:4d} tars  {k}/{m} members byte-identical to {QL_ROOT[sample_of(a)]}")
    print(f"\nmanifest: {out}")
    print(f"=== INTEGRITY GATE (group input archives are duplicates of a KEEP Q/L root) ===")
    if not bad:
        print(f"PASS -- {tot_k}/{tot_m} members across {len(rows)}/{len(rows)} archives")
        sys.exit(0)
    for r in bad[:20]:
        print(f"  !! {r[0]}/{r[1]}: {r[6]}")
    print(f"FAIL -- {len(bad)} archive(s) not fully accounted for; do NOT drop this class")
    sys.exit(1)
