#!/usr/bin/env python3
"""Thin dl_vtx_training/runs -- drop ALL *.pth, keep every other file, 2026-08-16.

Owner-confirmed after checking every one of the 20 large training arms
(2.9-4.9 GiB each) against its verdict in docs/pr/77-81: every trained arm is
REJECTED / NEGATIVE / inert / not deployed. Nothing from the entire pr/77-81
DL-vertex fine-tune campaign is in SBND production -- what shipped from it is
two knobs (dl_vtx_min_accept_score 4->10, dl_vtx_harvest recording-only),
neither of which reads a checkpoint. Specifically:
  ft0/ft1/ft1hn/ft1ps  round 1 (pr/77 sec.8e) -- "ft1hn === ft1 on every
                       out-of-fold metric" (hard-negative machinery inert);
                       round-1 verdict is why pr/81 exists ("gradients don't
                       pay at O(100)").
  ft2/ft2m3/ft2c9b*/ft2hn/ft2w  round 3 (pr/78) -- "ft2 is bit-inert: every
                       one of the 378 events identical to baseline".
  ft2u/ft2u-deploy     the one arm staged for deployment; pr/79 sec.3:
                       "REJECTED, -40/473 marginal live".
  hft1/hft1-deploy     pr/79 sec.11: "NEGATIVE, no live A/B, no flip".
  hr1/hr2/hr3/hr3-deploy  pr/81 round 2 -- hr1 FAIL, hr2 FAIL, hr3 "pass
                       (marginal)" on OOF but deploy screen -3, "nothing
                       ships now".

Every number those docs quote (config.json, eval.log, eval.tsv, oof_*.tsv,
lockbox.log, *.json, *.log, *.tsv, *.txt, *.png) is a non-.pth file and is
kept whole -- this script deletes ONLY files matching *.pth under
dl_vtx_training/runs, nothing else, in that tree, full stop.

Guard, modelled on thin_hubs_20260811.py's guard() chokepoint: every
candidate path must (a) resolve under ALLOWED_ROOT, (b) not be a symlink
(measured: 0 of 2195 .pth files are symlinks), (c) match *.pth exactly.

  python3 thin_dlruns_20260816.py           # dry run (default)
  CONFIRM=yes python3 thin_dlruns_20260816.py   # actually delete

Writes state-20260816/dlruns-removed.tsv (only under CONFIRM=yes).
"""
import fnmatch
import os
import sys

ROOT = "/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin"
ALLOWED_ROOT = os.path.join(ROOT, "dl_vtx_training", "runs")
STATE = os.environ.get("RETIRE_STATE",
                        os.path.join(ROOT, "scripts", "retire", "state-20260816"))
MANIFEST = os.path.join(STATE, "dlruns-removed.tsv")
CONFIRM = os.environ.get("CONFIRM", "no") == "yes"

if not os.path.isdir(ALLOWED_ROOT):
    sys.exit(f"REFUSING: {ALLOWED_ROOT} does not exist")

if CONFIRM and os.path.exists(MANIFEST):
    sys.exit(f"REFUSING: {MANIFEST} exists -- this round has already run (M13). "
              f"RETIRE_STATE=<new dir> to override.")


def guard(path):
    real = os.path.realpath(path)
    allowed_real = os.path.realpath(ALLOWED_ROOT)
    if not (real == allowed_real or real.startswith(allowed_real + os.sep)):
        raise SystemExit(f"REFUSING (outside ALLOWED_ROOT): {path}")
    if os.path.islink(path):
        raise SystemExit(f"REFUSING (symlink, never delete through one): {path}")
    if not fnmatch.fnmatch(os.path.basename(path), "*.pth"):
        raise SystemExit(f"REFUSING (not *.pth): {path}")
    return path


candidates = []
for cur, sub, files in os.walk(ALLOWED_ROOT):
    for f in files:
        if fnmatch.fnmatch(f, "*.pth"):
            candidates.append(os.path.join(cur, f))
candidates.sort()

n = len(candidates)
total = sum(os.path.getsize(p) for p in candidates if not os.path.islink(p))
print(f"dl_vtx_training/runs: {n} *.pth files, {total/2**30:.2f} GiB")
print(f"CONFIRM={'yes' if CONFIRM else 'no'}")

if not CONFIRM:
    print("\nfirst 10 (dry run):")
    for p in candidates[:10]:
        print(f"  would remove  {os.path.relpath(p, ROOT)}")
    print(f"  ... and {max(0, n - 10)} more")
    print("\ndry run only -- re-run with CONFIRM=yes to delete")
    sys.exit(0)

os.makedirs(STATE, exist_ok=True)
removed = 0
freed = 0
failed = []
with open(MANIFEST, "w") as fh:
    fh.write("# thin_dlruns_20260816.py\n")
    fh.write(f"# dl_vtx_training/runs *.pth-only sweep, {n} candidates\n")
    fh.write("path\tbytes\n")
    for p in candidates:
        try:
            guard(p)
            sz = os.path.getsize(p)
            os.remove(p)
            fh.write(f"{os.path.relpath(p, ROOT)}\t{sz}\n")
            removed += 1
            freed += sz
        except SystemExit as e:
            failed.append((p, str(e)))
        except OSError as e:
            failed.append((p, str(e)))

print(f"removed {removed}/{n} files, freed {freed/2**30:.2f} GiB")
if failed:
    print(f"!! {len(failed)} failures:")
    for p, why in failed[:20]:
        print(f"  !! {p}: {why}")
    sys.exit(1)
print(f"manifest: {MANIFEST}")
