#!/usr/bin/env python3
"""Inject the per-event trigger offset_us into existing opflash archives.

PDHD opflash archives produced before the trigger-offset plumbing landed carry
no `offset_us` in their tensorset metadata, so run_clus_evt.sh reads 0 and the
Q/L matching runs offset-free.  The offset never enters the light physics --
OpFlashFinder stamps it into the metadata only, never shifting the flash times
(OpFlashFinder.cxx:405 vs :442) -- so the flash/ophit tensors are identical
with or without it.  Re-running the full light reco just to write one metadata
scalar is wasted compute; this tool patches only that scalar in place.

For each work/<run6>_<evt>/opflash_pdhd-wct.tar.gz it:
  1. reads the DAQ event number from the tensorset metadata ("event"),
  2. derives the run from the work-dir name,
  3. computes offset_us from the ROOT trigoff/trigger_offset tree using the
     exact rule run_light_evt.sh uses (candidate nearest 250 us), and
  4. rewrites the tarball with offset_us added to the tensorset metadata,
     every other member (incl. the flash .npy arrays) copied verbatim.

Archives whose offset_us is already present and matching are skipped.

Usage:
  python3 restamp_offset.py [--dry-run] [-v] [GLOB ...]
    GLOB     archive globs (default: work/*/opflash_pdhd-wct.tar.gz)
    --dry-run  report what would change, write nothing
    -v         verbose (per-archive detail)
"""
import sys, os, re, io, json, glob, tarfile, tempfile, argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LIGHT_DIR = os.path.join(HERE, "input_data_7p8_new_coh_grouping")
TOL = 1e-6  # offset already considered stamped if |existing - computed| < TOL

# Per-ROOT-file cache of the trigoff tree, so a run's file is read once not
# once-per-event.
_trigoff_cache = {}


def find_root_file(run_padded):
    """ROOT light-data file for a run -- mirrors run_light_evt.sh:50-53."""
    rundir = os.path.join(LIGHT_DIR, "run%s" % run_padded)
    for pat in ("np04hd_raw_run%s_*.root" % run_padded, "*run%s*.root" % int(run_padded)):
        hits = sorted(glob.glob(os.path.join(rundir, pat)))
        if hits:
            return hits[0]
    return None


def load_trigoff(fn):
    if fn not in _trigoff_cache:
        import uproot
        _trigoff_cache[fn] = uproot.open(fn)["trigoff/trigger_offset"].arrays(library="np")
    return _trigoff_cache[fn]


def compute_offset(fn, run, event):
    """offset_us for (run, event) -- exact rule from run_light_evt.sh:76-80."""
    to = load_trigoff(fn)
    sel = (to["run"] == run) & (to["event"] == event)
    if not sel.any():
        return None
    i = np.argmin(np.abs(to["offset_us"][sel] - 250.0))
    return float(to["offset_us"][sel][i])


def read_tensorset_meta(tf):
    """Return (member_name, dict) for the tensorset metadata, or (None, None)."""
    for m in tf.getmembers():
        if "tensorset" in m.name and m.name.endswith("_metadata.json"):
            return m.name, json.loads(tf.extractfile(m).read())
    return None, None


def restamp(path, offset, verbose=False):
    """Rewrite `path` with offset_us added to every tensorset metadata member.
    All other members are copied byte-for-byte (same order, mode, mtime)."""
    # Write a sibling temp in the target's own directory so the final os.replace
    # is an atomic same-filesystem rename (the work dirs are on NFS; a /tmp temp
    # would fail cross-device).
    fd, tmp = tempfile.mkstemp(prefix=".restamp_", suffix=".tar.gz",
                               dir=os.path.dirname(path))
    os.close(fd)
    with tarfile.open(path) as src, tarfile.open(tmp, "w:gz") as dst:
        for m in src.getmembers():
            data = src.extractfile(m).read() if m.isfile() else b""
            if "tensorset" in m.name and m.name.endswith("_metadata.json"):
                md = json.loads(data)
                md["offset_us"] = offset
                data = json.dumps(md).encode()
                ti = tarfile.TarInfo(m.name)
                ti.size = len(data)
                ti.mode, ti.mtime = m.mode, m.mtime
                ti.uid, ti.gid = m.uid, m.gid
                ti.uname, ti.gname = m.uname, m.gname
                dst.addfile(ti, io.BytesIO(data))
            else:
                dst.addfile(m, io.BytesIO(data) if m.isfile() else None)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("globs", nargs="*", default=["work/*/opflash_pdhd-wct.tar.gz"])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    # Resolve each glob both relative to cwd and to the script dir, deduping by
    # real path so an archive matched two ways is processed once.
    seen, archives = set(), []
    for g in args.globs:
        for p in glob.glob(g) + glob.glob(os.path.join(HERE, g)):
            rp = os.path.realpath(p)
            if rp not in seen:
                seen.add(rp); archives.append(rp)
    archives.sort()
    if not archives:
        print("no archives matched", file=sys.stderr)
        return 1

    n_updated = n_skipped = n_missing = n_err = 0
    for path in archives:
        rel = os.path.relpath(path, HERE)
        m = re.search(r"work/(\d+)_", path)
        if not m:
            print("[skip] %s: cannot parse run from path" % rel); n_err += 1; continue
        run_padded = m.group(1)
        run = int(run_padded)

        with tarfile.open(path) as tf:
            mname, md = read_tensorset_meta(tf)
        if md is None:
            print("[skip] %s: no tensorset metadata" % rel); n_err += 1; continue
        event = md.get("event")
        existing = md.get("offset_us")

        root = find_root_file(run_padded)
        if root is None:
            print("[MISS] %s: no ROOT file for run %s" % (rel, run_padded)); n_missing += 1; continue
        try:
            offset = compute_offset(root, run, event)
        except Exception as e:
            print("[ERR ] %s: %s" % (rel, e)); n_err += 1; continue
        if offset is None:
            print("[MISS] %s: run=%d event=%s not in trigoff tree" % (rel, run, event))
            n_missing += 1; continue

        if existing is not None and abs(existing - offset) < TOL:
            if args.verbose:
                print("[ ok ] %s: offset_us=%.3f already present" % (rel, existing))
            n_skipped += 1; continue

        tag = "%.3f -> %.3f" % (existing, offset) if existing is not None else "%.3f" % offset
        if args.dry_run:
            print("[plan] %s: event=%s offset_us=%s" % (rel, event, tag))
        else:
            restamp(path, offset, args.verbose)
            print("[stmp] %s: event=%s offset_us=%s" % (rel, event, tag))
        n_updated += 1

    verb = "would update" if args.dry_run else "updated"
    print("\n%d %s, %d already-stamped, %d missing-offset, %d error  (of %d archives)"
          % (n_updated, verb, n_skipped, n_missing, n_err, len(archives)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
