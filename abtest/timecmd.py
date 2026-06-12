#!/usr/bin/env python3
"""Run a command, write 'rc/wall_s/maxrss_kb' to a meta file.

Usage: timecmd.py META_FILE CMD [ARGS...]
maxrss_kb = peak RSS over all waited-for descendants (getrusage CHILDREN).
"""
import resource
import subprocess
import sys
import time

meta = sys.argv[1]
t0 = time.time()
rc = subprocess.call(sys.argv[2:])
wall = time.time() - t0
ru = resource.getrusage(resource.RUSAGE_CHILDREN)
with open(meta, "w") as f:
    f.write(f"rc={rc}\nwall_s={wall:.0f}\nmaxrss_kb={ru.ru_maxrss}\n")
sys.exit(rc)
