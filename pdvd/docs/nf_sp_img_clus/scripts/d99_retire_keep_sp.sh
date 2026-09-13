#!/bin/bash
# doc pdvd/99 -- retire the PREVIOUS round's SP frames (owner 2026-09-13: "keep the latest SP results,
# and we can retire the previous round's SP result").  The previous round = the July (07-13, v6-wire)
# DNN-ROI frames in /home/xqian/pdvd-frame-store/<evt>_keep/, superseded by work/*_p98von.
#
#   d99_retire_keep_sp.sh                dry run
#   CONFIRM=1 d99_retire_keep_sp.sh      delete
#   NEGCTL=1  d99_retire_keep_sp.sh      negative control: inject an out-of-scope path -> must refuse (rc 13)
#
# Scope: ONLY /home/xqian/pdvd-frame-store/<run6>_<idx>_keep/protodune-sp-dnnroi-frames-anode[0-7].tar.bz2
# and the symlinks under pdvd/work whose one-hop target is one of those files or is a link that is
# (work/<evt>_keep/<frame> -> store; work/<evt>_d27fresh/<frame> -> ../../work/<evt>_keep/<frame>; any
# other arm found by the census).  work/<evt>_keep itself (imaging, calib, mabc, logs = doc pdvd/24's
# "latest production data") and every non-frame file are untouched.
#
# KEPT events = every event a committed script or doc Repro reads these frames for (census 2026-09-13):
#   039252_0..17  doc qlmatch/18 Repro (039252_${i}_keep frames, i = 0..17), check_clus97_tail_waveforms.py
#                 (039252_0), doc nf_sp_img_clus/28 (039252_2), d29_gain_recomb.py (039252_0 via d27fresh)
#   039253_0      d29_gain_recomb.py (via d27fresh)
#   039349_81 039349_4 039349_15 039349_68 039349_2
#                 this doc's own reproduction controls: p98kchk (imaging) links these July frames directly and p98kq
#                 (clustering + PR) is staged from d27fresh's imaging of them; keeping them keeps both controls rerunnable
#                 (039252_0, 039252_13, 039253_0 are the other three control events and are already kept above)
# cathode_tail_waveform_check.py takes run/idx on the command line and is not bound to an event.
#
# Guards, in order: KEPT-list census re-run (a literal event id cited in either repo but not KEPT ->
# rc 12); liveness (wire-cell running or a process holding a store file open -> rc 11); scope regex on
# every path (rc 13); manifest written BEFORE anything is removed (file sha256 + bytes of every archive,
# every link and its target); post-state dangling-link census must add no link outside the removed set.
set -u
IMG=/nfs/data/1/xqian/toolkit-dev/wcp-porting-img
STORE=/home/xqian/pdvd-frame-store
WORK=$IMG/pdvd/work
OUT=$IMG/pdvd/docs/nf_sp_img_clus/d99
KEPT=" 039253_0 039349_81 039349_4 039349_15 039349_68 039349_2 $(for i in $(seq 0 17); do printf '039252_%s ' $i; done)"
mkdir -p "$OUT"

# ---- 1. census: literal cited events must be KEPT -------------------------------------------
cited=$(grep -rhoI --exclude-dir=work --exclude-dir=.git --exclude-dir=archive \
          -E "0392(52|53|349)?_[0-9]+_(keep|d27fresh)/protodune-sp-dnnroi" "$IMG" 2>/dev/null \
        | grep -oE "^0[0-9]{5}_[0-9]+" | sort -u)
for e in $cited; do
    case "$KEPT" in *" $e "*) ;; *) echo "REFUSE census: $e is cited but not KEPT" >&2; exit 12 ;; esac
done
echo "census: cited literal events [$(echo $cited)] all KEPT"

# ---- 2. targets ------------------------------------------------------------------------------
arch=(); bytes=0
for f in "$STORE"/*_keep/protodune-sp-dnnroi-frames-anode*.tar.bz2; do
    e=$(basename "$(dirname "$f")"); e=${e%_keep}
    case "$KEPT" in *" $e "*) continue ;; esac
    arch+=("$f"); bytes=$((bytes + $(stat -c%s "$f")))
done
[ "${NEGCTL:-0}" = 1 ] && arch+=("$WORK/039252_0_p98von/protodune-sp-dnnroi-frames-anode0.tar.bz2") && echo "NEGCTL: injected an ON-arm frame"
for f in "${arch[@]}"; do
    [[ "$f" =~ ^/home/xqian/pdvd-frame-store/0[0-9]{5}_[0-9]+_keep/protodune-sp-dnnroi-frames-anode[0-7]\.tar\.bz2$ ]] \
        || { echo "REFUSE scope: $f" >&2; exit 13; }
done
# links whose one-hop target is a retired archive or a link already collected -- iterated to a FIXED
# POINT (keep -> store, d27fresh -> keep, and any arm that links through d27fresh), resolved lexically
declare -A RET=(); for f in "${arch[@]}"; do RET["$f"]=1; done
links=()
mapfile -t ALL < <(find "$WORK" -maxdepth 2 -type l -name 'protodune-sp-dnnroi-frames-anode*.tar.bz2')
prev=-1; pass=0
while [ "${#links[@]}" != "$prev" ]; do
    prev=${#links[@]}; pass=$((pass+1))
    for l in "${ALL[@]}"; do
        [ -n "${RET[$l]:-}" ] && continue
        t=$(readlink "$l"); case "$t" in /*) abs=$t ;; *) abs=$(cd "$(dirname "$l")" && realpath -s -m "$t") ;; esac
        abs=${abs/#\/home\/xqian\/toolkit-dev\/wcp-porting-img/$IMG}
        if [ -n "${RET[$abs]:-}" ]; then RET["$l"]=1; links+=("$l"); fi
    done
done
echo "link closure: ${#links[@]} links after $pass passes"
echo "retire: ${#arch[@]} archives, $(awk -v b=$bytes 'BEGIN{printf "%.2f", b/2^30}') GiB; ${#links[@]} links; kept events: $(echo $KEPT | wc -w)"

# ---- 3. liveness -------------------------------------------------------------------------------
[ "$(pgrep -c '^wire-cell')" = 0 ] || { echo "REFUSE liveness: wire-cell running" >&2; exit 11; }
if command -v fuser >/dev/null && fuser -s "${arch[@]:0:50}" 2>/dev/null; then echo "REFUSE liveness: store file open" >&2; exit 11; fi

if [ "${CONFIRM:-0}" != 1 ]; then echo "dry run (set CONFIRM=1 to delete)"; exit 0; fi

# ---- 4. manifest first -------------------------------------------------------------------------
M=$OUT/retire_keep_sp_manifest.txt
{
  echo "# doc pdvd/99: July (_keep, v6-wire) SP frames retired $(date -Is); file sha256 bytes path"
  printf '%s\n' "${arch[@]}" | xargs -P 16 -n 1 sh -c 'echo "$(sha256sum < "$1" | cut -c1-64) $(stat -c%s "$1") $1"' _ | sort -k3
  echo "# links removed: link -> target"
  for l in "${links[@]}"; do echo "$l -> $(readlink "$l")"; done
} > "$M"
[ "$(grep -c '^[0-9a-f]\{64\} ' "$M")" = "${#arch[@]}" ] || { echo "REFUSE: manifest incomplete" >&2; exit 14; }
before=$(find "$WORK" -maxdepth 2 -xtype l | wc -l)

# ---- 5. delete -----------------------------------------------------------------------------------
for l in "${links[@]}"; do rm -f -- "$l"; done
for f in "${arch[@]}"; do rm -f -- "$f"; done
after=$(find "$WORK" -maxdepth 2 -xtype l | wc -l)
echo "deleted ${#arch[@]} archives + ${#links[@]} links; dangling links in work before $before after $after (must be equal)"
echo "manifest $M"
[ "$before" = "$after" ] || exit 16
