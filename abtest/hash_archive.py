#!/usr/bin/env python3
"""Content-hash WCT output archives, ignoring embedded timestamps.

tar members are written with mtime=time(0) (custard.hpp) and zip members
carry wall-clock timestamps, so raw archive bytes are never reproducible.
This hashes sha256(member_name + payload) over members sorted by name.

Usage: hash_archive.py [--members] ARCHIVE [ARCHIVE...]
Output (default):  <rollup-sha256>  <n-members>  <archive-path>
With --members, per-member lines are printed first, indented.
"""
import hashlib
import sys
import tarfile
import zipfile


def members(path):
    """Yield (name, payload-bytes) for regular-file members, sorted by name."""
    if path.endswith(".zip"):
        with zipfile.ZipFile(path) as zf:
            for name in sorted(zf.namelist()):
                if name.endswith("/"):
                    continue
                yield name, zf.read(name)
    else:
        with tarfile.open(path) as tf:
            infos = {m.name: m for m in tf.getmembers() if m.isfile()}
            for name in sorted(infos):
                yield name, tf.extractfile(infos[name]).read()


def main():
    args = sys.argv[1:]
    show_members = False
    if args and args[0] == "--members":
        show_members = True
        args = args[1:]
    if not args:
        sys.exit(__doc__)
    for path in args:
        rollup = hashlib.sha256()
        count = 0
        try:
            for name, payload in members(path):
                h = hashlib.sha256(name.encode() + payload).hexdigest()
                rollup.update(h.encode())
                count += 1
                if show_members:
                    print(f"  {h}  {name}")
        except Exception as exc:  # noqa: BLE001 - report and continue
            print(f"ERROR  -  {path}  ({exc})")
            continue
        print(f"{rollup.hexdigest()}  {count}  {path}")


if __name__ == "__main__":
    main()
