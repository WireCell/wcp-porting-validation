#!/usr/bin/env python3
"""Thin dl_vtx_training/runs -- drop ALL *.pth, keep every other file, 2026-08-17.

Fork of thin_dlruns_20260816.py, same guard/CONFIRM shape. That round dropped
all 2195 .pth from the pr/77-81 campaign; this round's 648 .pth (17.4 GiB,
28,770,937 bytes each) are ALL from doc pr/89's Arm B round-4 retrain,
created 2026-08-16 17:09-20:44 (i.e. the entire regrowth since 08-16), in six
arms: hr4, hr4b, hr4-hum, hr4b-hum, hr4-maxa, hr4b-maxa (108 .pth each).

Verdicts, both halves of the split:
  hr4/hr4-hum/hr4-maxa   the FIRST run, NaN-poisoned (pr/89 sec.11.8): "17 of
                         the 18 folds died at epoch 0 with loss=nan ... one
                         nan backward() then poisons every weight
                         irreversibly". On disk hr4-maxa's best.json reports
                         best_epoch=0 on all six folds -- the weights encode
                         nothing.
  hr4b/hr4b-hum/hr4b-maxa  the CLEAN rerun, CLOSED NEGATIVE (pr/89 sec.11.11):
                         "guard-predicted delta <= 0 in 125/126 cells ...
                         reject->ACCEPT = 0 in every cell ... no lockbox read
                         spent on Arm B; Arm B contributes nothing to the
                         live A/B." Round table (sec.12): "B retrain |
                         NEGATIVE ... | nothing". Production is unchanged by
                         this round.

Production reads NEITHER: SBND's dl_weights is
'uboone/scn_vtx/t48k-m16-l5-lr5d-res0.5-CP24.pth', resolving into
wire-cell-data/, entirely outside this tree (cfg/pgrapher/experiment/sbnd/
{clus,wct-pr-perevt}.jsonnet). Reverse grep for any config/runner path into
dl_vtx_training: zero hits.

Every number those docs quote (config.json, best.json, *.log, *.tsv, *.json)
is a non-.pth file and is kept whole -- this script deletes ONLY files
matching *.pth under dl_vtx_training/runs, nothing else, in that tree, full
stop. dl_vtx_training/data/ (58M, 18 frozen snapshots backing docs pr/77-81
and pr/88) is untouched -- it is outside ALLOWED_ROOT entirely.

Guard, identical to 08-16's: every candidate path must (a) resolve under
ALLOWED_ROOT, (b) not be a symlink, (c) match *.pth exactly.

  python3 thin_dlruns_20260817.py           # dry run (default)
  CONFIRM=yes python3 thin_dlruns_20260817.py   # actually delete

Writes state-20260817/dlruns-removed.tsv (only under CONFIRM=yes).
"""
import fnmatch
import os
import sys

ROOT = "/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin"
ALLOWED_ROOT = os.path.join(ROOT, "dl_vtx_training", "runs")
STATE = os.environ.get("RETIRE_STATE",
                        os.path.join(ROOT, "scripts", "retire", "state-20260817"))
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
    fh.write("# thin_dlruns_20260817.py\n")
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
