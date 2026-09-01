# prod-2026-09-01b — the SBND production operating point, pinned (doc 87)

Supersedes `ref/prod-2026-09-01` (doc 77 round 4).  `prod_cfg_gate.py` with no
`--ref` uses the **newest** `ref/prod-*`, so this one is now the default.

## What moved, and on whose word

Exactly one key, recorded by the old reference before it was superseded:

```
DRIFT     : prod_prjob.json
  ADDED   [24].data.save_in_scope = True
```

`save_in_scope` puts the **in-scope cluster set** — plus the per-bundle summary
(flags, flash, npoints, length) — into a `T_cluster` tree in
`tracking-pr.root`.  Owner ask, 2026-09-01:

> *"Can we fix this problem, I think we want the final rootfile for each event
> to contain this information?"*

The C++ default is still `false` and stays pinned by
`root/test/doctest_sbnd_pr_tracking_defaults.cxx`.  This is the SBND **job's**
operating point, not a default change.

Full rationale, acceptance and gates: `docs/87_production-output-minimization.md`
§4.  Acceptance was `set(T_cluster.cluster_id where in_scope) ==
parse_prbee(mabc-pr.zip)` **exactly on 308/308 events**, and the knob-ON
difference is one added tree with every pre-existing tree identical on 308/308.

## CONSEQUENCE — read this before running a ROOT gate across the epoch

`tracking-pr.root` now carries a tree that no pre-doc-87 arm has, so

```
scripts/pr94_root_gate.py <pre-doc-87 arm> <new arm>      # will FAIL on the tree list
```

That FAIL is expected and is not a regression.  Use

```
scripts/pr87_root_tree_diff.py <armA> <armB>
```

which compares every **shared** tree (NaN-aware) and names extra trees
separately.  Arms produced from this point on compare to each other normally.

## Contents and use

Same shape as `prod-2026-09-01`: `consumers.sha256` (21 compiled artifacts),
`prod_prjob.json` (the SBND PR job in full), and the committed 308-event gate
manifests `gate308-{mcp1k,nuecc48,ncpi0}.txt`.

```bash
scripts/cfg/prod_cfg_gate.py                    # rc 0 = matches; rc 1 = drift
scripts/cfg/prod_cfg_gate.py --refresh          # ONLY on an intended flip
```

A refresh with no record in the round doc of which knob moved and on whose word
is indistinguishable from the drift this reference exists to catch.
