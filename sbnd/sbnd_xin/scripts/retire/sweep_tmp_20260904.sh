#!/bin/bash
# doc 100 / doc pdvd/29 -- sweep ~/tmp, 2026-09-04.
#
#   ./scripts/retire/sweep_tmp_20260904.sh              # dry run (default)
#   CONFIRM=yes ./scripts/retire/sweep_tmp_20260904.sh  # actually delete
#
# OWNER SCOPE: "Please also clean up a bit the ~/tmp directory."  132 GiB.
#
# WHAT THIS SWEEP DELETES, AND WHAT IT NEVER TOUCHES.
# ~/tmp is overwhelmingly PINNED LIBRARY SNAPSHOTS (1.2-1.9 GiB copies of
# local/lib taken so a campaign survives a peer's mid-round `wcbuild`) and
# cmake build trees.  The actual records -- the arm logs, gate outputs, .md
# appendices, the config dumps -- are small text and are KEPT IN PLACE.  This
# sweep removes only whole lib*/ snapshot dirs and build trees, never a *.log,
# *.txt, *.md, *.json or *.zip.  A libsnap is regenerable from the commit its
# doc records; a gate log is not.
#
# THE RULE (doc 98): A PINNED BINARY GOES WITH ITS ARMS, NEVER BEFORE.
# Re-derived per pin for this round -- which doc names it, which work dirs that
# doc's arms are, and whether any of those arms survive the 09-04 retire:
#
#   DROP  doc28/lib_* (17 snaps)  doc 28 rounds 1-3 CLOSED; every d28* arm except
#                                 the d28dlfp symlink spine retires this round
#   DROP  d31{,r3,r4,r5,r7}lib    doc 31 CLOSED; the d31* arms retire
#   DROP  d92gate-libsnap         work-*-d92gate{,pr} retire
#   DROP  d99-libsnap             work-*-d99fix{,pr} retire
#   DROP  doc94c-libsnap          backs work-*-r2scan*, which retire
#   DROP  doc94r3b-libsnap        backs work-dbg25a-*, which retire
#   DROP  d99r2-cmake2            a cmake build tree, regenerable (doc 98
#                                 released three of exactly this class)
#   DROP  doc37/cmbuild           ditto, 5.1 GiB
#   DROP  m1gate                  zero citations, old scratch
#
#   KEEP  d39/lib_d39             *** LIVE ***  doc pdvd/39 sec 0 names it
#   KEEP  doc25gate/ doc25r12/ doc25r13/ pinlib*   *** LIVE PEER ***
#   KEEP  d97b-libsnap            the SBND PRODUCTION binary (work-*-d97fv*)
#   KEEP  d99r2-libsnap           backs work-*-d99r3prod{,pr}, which SURVIVE --
#                                 the ref/prod-2026-09-05 operating point
#   KEEP  prod0901b-libsnap       backs work-em114*, which survive
#   KEEP  d31r6lib                backs 039252_*_d31r6e2e, kept as substrate
#   KEEP  doc37/lib_{pin,base,new}  doc 37 sec 0 and line 650 cite these by md5
#                                 as the symbol-present/absent witness pair, and
#                                 doc 37's gate arms (d37on05/off0/off1) SURVIVE
#   KEEP  doc87/lib-*             doc 87's arms are PROTECTED.txt pins whose
#                                 written release condition is not yet met
#
# CLAUDE SESSION SCRATCHPADS: kept for every LIVE session.  AGE IS NOT
# LIVENESS -- claude-25225/.../7117f9b1-… is 18 GiB and had not been written
# since 09-03 05:49, which reads as dead; `claude --resume 7117f9b1-…` is
# RUNNING.  Liveness is derived from ps, never from mtime.
set -u
TMP=/home/xqian/tmp
CONFIRM=${CONFIRM:-no}

# --- interlock A: the arms these pins back must actually be gone (or planned).
# A pin dropped BEFORE its arms leaves a live arm with no binary to re-run it.
SX=/nfs/data/1/xqian/toolkit-dev/wcp-porting-img/sbnd/sbnd_xin
if [ -d "$SX/work-mcp1k-d92gate" ] || [ -d "$SX/work-mcp1k-d99fix" ]; then
  echo "NOTE: the sbnd arms these pins back are still on disk."
  echo "      That is fine for a DRY RUN, but run retire_20260904.sh first."
  [ "$CONFIRM" = "yes" ] && { echo "REFUSE: pins must go WITH their arms, not before."; exit 2; }
fi

# --- interlock B: never sweep a live session's scratchpad
LIVE=$(ps -u "$(id -un)" -o args= | grep -oE 'claude --resume [0-9a-f]{8}-[0-9a-f-]+' | awk '{print $3}' | sort -u)
echo "== live claude sessions (scratchpads protected) =="
echo "${LIVE:-  (none)}" | sed 's/^/  /'

DROP=""
for d in doc28/lib_base doc28/lib_base2 doc28/lib_l1 doc28/lib_l2 doc28/lib_post2 \
         doc28/lib_dump2B doc28/lib_dump2P doc28/lib_dump3B doc28/lib_dump3P \
         doc28/lib_dumpB doc28/lib_dumpP doc28/lib_r2base doc28/lib_r2post \
         doc28/lib_r2post3 doc28/lib_r2post5 doc28/lib_r3fix doc28/lib_r3post \
         doc28/lib_utilonly \
         d31lib d31r3lib d31r4lib d31r5lib d31r7lib \
         d92gate-libsnap d99-libsnap doc94c-libsnap doc94r3b-libsnap \
         d99r2-cmake2 doc37/cmbuild m1gate; do
  [ -e "$TMP/$d" ] && DROP="$DROP $d"
done

# --- interlock C: nothing on the DROP list may be a live scratchpad, a KEEP
# pin, or anything other than a directory of libraries / build output.
for d in $DROP; do
  case "$d" in
    *claude-25225*|*d39*|*doc25*|*pinlib*|*d97b-libsnap*|*d99r2-libsnap*|\
    *prod0901b-libsnap*|*d31r6lib*|*doc87*|*lib_pin*|*lib_base|*lib_new)
      case "$d" in doc28/lib_base) ;; *) echo "REFUSE: $d is a KEEP pin"; exit 2;; esac;;
  esac
  [ -d "$TMP/$d" ] || { echo "REFUSE: $TMP/$d is not a directory"; exit 2; }
done

echo
echo "== sweep set =="
for d in $DROP; do printf "  %-24s %s\n" "$d" "$(du -sh "$TMP/$d" 2>/dev/null | cut -f1)"; done
TOT=$(du -sc --block-size=1 $(for d in $DROP; do echo "$TMP/$d"; done) 2>/dev/null | tail -1 | cut -f1)

if [ "$CONFIRM" != "yes" ]; then
  echo
  printf "DRY RUN -- nothing deleted.  Would free %.2f GiB from %d dirs.\n" \
      "$(echo "$TOT/1073741824" | bc -l)" "$(echo $DROP | wc -w)"
  echo "~/tmp now: $(du -sh $TMP 2>/dev/null | cut -f1)"
  echo
  echo "To execute:  CONFIRM=yes ./scripts/retire/sweep_tmp_20260904.sh"
  exit 0
fi

echo "== DELETING =="
for d in $DROP; do rm -rf -- "$TMP/$d" && echo "  removed $d"; done
echo
echo "~/tmp now: $(du -sh $TMP 2>/dev/null | cut -f1)"
