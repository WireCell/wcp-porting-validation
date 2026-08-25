# Imported by site.py at interpreter startup, BEFORE anything the process
# imports.  Puts this dir at sys.path[0] so the SCN_Vertex wrapper next to it
# shadows toolkit/pyutil/python/SCN_Vertex.py, which run_pr_chain_batch.sh
# prepends to PYTHONPATH.  Only active for processes that carry this dir on
# PYTHONPATH, i.e. only the probe runs.
import os, sys
_d = os.path.dirname(os.path.abspath(__file__))
if os.environ.get("WCT_SCN_PROBE") and _d in sys.path:
    sys.path.remove(_d)
    sys.path.insert(0, _d)
